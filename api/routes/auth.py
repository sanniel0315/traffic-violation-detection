#!/usr/bin/env python3
"""簡易帳號登入 API"""
from __future__ import annotations

from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.models import User, get_db, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["Auth"])

AUTH_COOKIE = "tvd_session"
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-this-auth-secret")
AUTH_TTL_HOURS = int(os.getenv("AUTH_TTL_HOURS", "24"))


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"
    enabled: bool = True


class UserUpdateRequest(BaseModel):
    role: str | None = None
    enabled: bool | None = None


class UserPasswordUpdateRequest(BaseModel):
    password: str


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _unb64url(data: str) -> bytes:
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("utf-8"))


def _sign(payload: str) -> str:
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(sig)


def make_token(username: str, role: str) -> str:
    exp = int((datetime.utcnow() + timedelta(hours=AUTH_TTL_HOURS)).timestamp())
    payload = _b64url(json.dumps({"u": username, "r": role, "exp": exp}).encode("utf-8"))
    return f"{payload}.{_sign(payload)}"


def parse_token(token: str) -> dict | None:
    try:
        payload, sig = str(token).split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        data = json.loads(_unb64url(payload).decode("utf-8"))
        if int(data.get("exp", 0)) < int(datetime.utcnow().timestamp()):
            return None
        return data
    except Exception:
        return None


def _set_cookie(response: Response, token: str):
    response.set_cookie(
        AUTH_COOKIE,
        token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=AUTH_TTL_HOURS * 3600,
        path="/",
    )


def _clear_cookie(response: Response):
    response.delete_cookie(AUTH_COOKIE, path="/")


def _current_user(request: Request, db: Session) -> User | None:
    token = request.cookies.get(AUTH_COOKIE)
    if not token:
        return None
    data = parse_token(token)
    if not data:
        return None
    username = str(data.get("u") or "")
    if not username:
        return None
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.enabled:
        return None
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="未登入")
    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if str(current_user.role or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="需要管理者權限")
    return current_user


@router.post("/login")
def login(data: LoginRequest, response: Response, db: Session = Depends(get_db)):
    username = str(data.username or "").strip()
    if not username or not data.password:
        raise HTTPException(status_code=400, detail="帳號密碼不可為空")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.enabled:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    token = make_token(user.username, user.role or "viewer")
    _set_cookie(response, token)
    return {"status": "success", "username": user.username, "role": user.role}


@router.post("/logout")
def logout(response: Response):
    _clear_cookie(response)
    return {"status": "success"}


@router.get("/me")
def me(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="未登入")
    return {"status": "success", "username": user.username, "role": user.role}


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    items = (
        db.query(User)
        .order_by(User.id.asc())
        .all()
    )
    return {
        "status": "success",
        "items": [
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "enabled": bool(u.enabled),
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "updated_at": u.updated_at.isoformat() if u.updated_at else None,
            }
            for u in items
        ],
    }


@router.post("/users")
def create_user(
    data: UserCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    username = str(data.username or "").strip()
    password = str(data.password or "")
    role = str(data.role or "viewer").lower().strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="帳號與密碼不可為空")
    if role not in ("admin", "ops", "viewer"):
        raise HTTPException(status_code=400, detail="角色不合法")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="帳號已存在")
    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        enabled=bool(data.enabled),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "status": "success",
        "item": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "enabled": bool(user.enabled),
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        },
    }


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    data: UserUpdateRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="找不到使用者")
    if data.role is not None:
        role = str(data.role or "").lower().strip()
        if role not in ("admin", "ops", "viewer"):
            raise HTTPException(status_code=400, detail="角色不合法")
        user.role = role
    if data.enabled is not None:
        if user.id == admin.id and not bool(data.enabled):
            raise HTTPException(status_code=400, detail="不可停用目前登入的管理者")
        user.enabled = bool(data.enabled)
    # 至少保留一位啟用中的 admin
    if str(user.role).lower() != "admin" or not bool(user.enabled):
        active_admins = (
            db.query(User)
            .filter(User.role == "admin", User.enabled == True, User.id != user.id)
            .count()
        )
        if active_admins <= 0:
            raise HTTPException(status_code=400, detail="系統至少需要一位啟用中的管理者")
    db.commit()
    db.refresh(user)
    return {
        "status": "success",
        "item": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "enabled": bool(user.enabled),
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        },
    }


@router.put("/users/{user_id}/password")
def update_user_password(
    user_id: int,
    data: UserPasswordUpdateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="找不到使用者")
    password = str(data.password or "")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密碼長度至少 6 碼")
    user.password_hash = hash_password(password)
    db.commit()
    return {"status": "success"}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="找不到使用者")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="不可刪除目前登入的管理者")
    if str(user.role).lower() == "admin" and bool(user.enabled):
        active_admins = (
            db.query(User)
            .filter(User.role == "admin", User.enabled == True, User.id != user.id)
            .count()
        )
        if active_admins <= 0:
            raise HTTPException(status_code=400, detail="系統至少需要一位啟用中的管理者")
    db.delete(user)
    db.commit()
    return {"status": "success"}
