"""signal(號誌操作員)角色:後端要接受,前端權限矩陣要把它限制在動態號誌控制。

使用者需求:「用使用者登入後就只看到動態號誌所有功能」。
後端只管角色合法性;能看到哪些頁由前端 permissionMatrix 決定,
所以兩邊都要驗,少一邊都會變成「建得出帳號但登入後一片空白」或「看得到全部」。
"""
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mem_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from api.models import Base
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng)()


def test_backend_accepts_signal_role_and_rejects_unknown():
    from fastapi import HTTPException
    from api.routes import auth as A
    from api.models import User
    db = _mem_session()
    admin = User(username="root", password_hash=A.hash_password("x"), role="admin", enabled=True)
    db.add(admin); db.commit(); db.refresh(admin)

    ok = A.create_user(A.UserCreateRequest(username="sig1", password="pw", role="signal", enabled=True),
                       db=db, _admin=admin)
    assert ok["item"]["role"] == "signal"

    with pytest.raises(HTTPException) as e:
        A.create_user(A.UserCreateRequest(username="bad", password="pw", role="operator", enabled=True),
                      db=db, _admin=admin)
    assert e.value.status_code == 400

    # 改角色也走同一份清單
    u = db.query(User).filter(User.username == "sig1").first()
    out = A.update_user(u.id, A.UserUpdateRequest(role="viewer"), db=db, admin=admin)
    assert out["item"]["role"] == "viewer"
    assert "signal" in A.VALID_ROLES


def test_frontend_matrix_confines_signal_role_to_signal_hub():
    """前端權限矩陣:signal 角色只有 'signal' 一個鍵;signal_hub 對到它;
    側欄改用 hasPerm('signal') 而不是 isAdmin;登入落地到 signal_hub。"""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "signal:['signal','monitor','lock','io']," in html, "signal 角色預設應為 動態號誌+即時監控+電子鎖+I/O"
    assert "signal_hub:'signal'," in html, "pagePermMap 少了 signal_hub"
    assert "lock:'lock'," in html and "io_panel:'io'," in html, "電子鎖 / I/O 沒有權限鍵"
    assert "v-if=\"hasPerm('lock')\" class=\"menu-item\"" in html and "v-if=\"hasPerm('io')\" class=\"menu-item\"" in html
    assert "{key:'signal',label:'動態號誌控制'}" in html, "permissionCatalog 少了 signal 鍵"
    assert '''v-if="hasPerm('signal')" class="menu-item" :class="{active:page==='signal_hub'}"''' in html, \
        "側欄的動態號誌控制仍綁 isAdmin"
    # 2026-09-06 使用者:「登入就是動態號誌頁面」—— 不再只有 signal 角色落地在這裡,
    # 任何有 signal 權限的帳號登入都直接進動態號誌頁。
    assert "if(hasPerm('signal')) return 'signal_hub';" in html
    # 四個角色選單都要有 signal 可選
    assert html.count('<el-option label="signal（號誌操作員）" value="signal"/>') == 4
    # 儲存 / 載入 / 重設 三處都要帶 signal,少一處重新整理就會掉回預設
    for frag in ("signal:permissionMatrix.signal||[],",
                 "signal:Array.isArray(raw.signal)?raw.signal:[...defaultPermissionMatrix.signal],",
                 "permissionMatrix.signal=[...defaultPermissionMatrix.signal];"):
        assert frag in html, frag
