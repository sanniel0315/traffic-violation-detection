"""續約要對齊時間錨點,不可以「做完事再睡固定秒數」。

🛑 2026-09-17 現場:續約設 10 秒,實際間隔卻是 10/17/10/18/10/19… 交替,
   285 則裡 50 次超過 15 秒;而部署後 4 次掉回定時控制有 3 次就落在這種空窗
   (距上次續約 16.5 / 19.7 / 20.6 秒)。原因是迴圈把工作耗時累加到週期上。
"""
import os
import re
import pathlib

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")

SRC = pathlib.Path("api/routes/signal_tc3.py").read_text(encoding="utf-8")


def test_loop_waits_on_anchor_not_fixed_sleep():
    """迴圈末端不可以是 wait(AUTH_RENEW_SEC) —— 那就是會累加的寫法。"""
    body = SRC[SRC.index("def _auth_renew_loop"):SRC.index("def start_auth_renew")]
    assert "_next_at += AUTH_RENEW_SEC" in body, "要用時間錨點推進"
    assert not re.search(r"shutdown_event\.wait\(AUTH_RENEW_SEC\)", body), \
        "睡固定秒數會把工作耗時累加成空窗"


def test_slow_round_is_counted():
    """落後要被記下來,不然下次又只能靠猜。"""
    body = SRC[SRC.index("def _auth_renew_loop"):SRC.index("def start_auth_renew")]
    assert "late_n" in body


def test_anchor_resets_when_far_behind():
    """落後超過一個週期就重設錨點 —— 不可以補送一串命令灌爆序列線。"""
    body = SRC[SRC.index("def _auth_renew_loop"):SRC.index("def start_auth_renew")]
    assert "_next_at = time.time() + AUTH_RENEW_SEC" in body
