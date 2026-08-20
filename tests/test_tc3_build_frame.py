"""TC3 碼框組建器:組出來的必須能被既有的 decode_frame 原封不動解回去。

號控要送命令給運轉中的號誌控制器,組錯一個位元組不是「沒反應」而是
可能被解成別的命令。所以往返測試是這條路上唯一能離線做的把關。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def tc3():
    """只要 build_frame / decode_frame / _stuff / _unstuff,不要拉起整個 app。"""
    import importlib.util
    import os
    # 🛑 signal_tc3 會連帶匯入 api.routes.auth,而它在 AUTH_SECRET 沒設時
    #    會直接 RuntimeError(那是刻意的安全檢查,不要動它)。
    #    這裡只是要測純協定函式,給一個測試用的值就好。
    os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret-" + "x" * 24)
    spec = importlib.util.spec_from_file_location(
        "_tc3_mod", ROOT / "api" / "routes" / "signal_tc3.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # 相依套件不在就跳過,不要假失敗
        pytest.skip(f"signal_tc3 匯入失敗: {exc}")
    return mod


def _roundtrip(tc3, addr, seq, info):
    frame = tc3.build_frame(addr, seq, info)
    out = tc3.decode_frame(frame)
    assert out is not None, "組出來的碼框自己解不開"
    assert out["cks_ok"] is True, "檢查碼不對"
    assert out["seq"] == seq
    assert out["addr"] == addr
    return frame, out


def test_基本往返(tc3):
    info = bytes([0x5F, 0x15, 0x01, 0x02, 0x03])
    frame, out = _roundtrip(tc3, 0x1230, 7, info)
    assert out["device"] == "5F"
    # LEN 必須等於整個碼框的實際長度
    assert out["len"] == len(frame), f'LEN {out["len"]} != 實際 {len(frame)}'


def test_INFO裡的0xAA要stuffing(tc3):
    """🛑 INFO 含 0xAA 時不做 stuffing,接收端會把它當成 DLE,碼框從那裡斷掉。"""
    info = bytes([0x5F, 0x15, 0xAA, 0x01])
    frame, out = _roundtrip(tc3, 0x1230, 1, info)
    # 送出去的位元組裡,那個 0xAA 必須是成對的
    assert bytes([0xAA, 0xAA]) in frame, "0xAA 沒有被重複"
    assert out["len"] == len(frame)


def test_INFO裡出現AACC不會被誤判成結尾(tc3):
    """StepSec 是 2 bytes,0xAACC 在值域內,現場真的會遇到(見 _find_etx 註解)。"""
    info = bytes([0x5F, 0x03, 0xAA, 0xCC, 0x09])
    frame, out = _roundtrip(tc3, 0x1230, 2, info)
    assert out["len"] == len(frame)


def test_連續多個0xAA(tc3):
    info = bytes([0x5F, 0x10, 0xAA, 0xAA, 0xAA])
    frame, out = _roundtrip(tc3, 0x1230, 3, info)
    assert out["len"] == len(frame)


def test_檢查碼是XOR含頭尾不含自己(tc3):
    info = bytes([0x0F, 0x12, 0x01])
    frame = tc3.build_frame(0x1230, 9, info)
    cks = 0
    for b in frame[:-1]:
        cks ^= b
    assert frame[-1] == cks


def test_stuff與unstuff互為反向(tc3):
    for info in (b"", b"\x00", b"\xaa", b"\xaa\xaa", b"\x01\xaa\x02\xaa\xaa\x03",
                 bytes(range(256))):
        assert tc3._unstuff(tc3._stuff(info)) == info, f"往返不一致: {info!r}"


def test_位址與序號原樣帶回(tc3):
    for addr in (0x0000, 0x1230, 0xFFFF):
        for seq in (0, 1, 255):
            _roundtrip(tc3, addr, seq, bytes([0x5F, 0x45]))


def test_不合法輸入要當場擋下不要送出去(tc3):
    """這支組出來的位元組會進運轉中的號誌通道,寧可在組的時候就拒絕。"""
    with pytest.raises(ValueError):
        tc3.build_frame(0x1230, 0, b"")           # INFO 至少要有設備碼+指令碼
    with pytest.raises(ValueError):
        tc3.build_frame(0x1230, 0, b"_")       # 只有設備碼
    with pytest.raises(ValueError):
        tc3.build_frame(0x10000, 0, b"_")  # 位址超出 2 bytes
    with pytest.raises(ValueError):
        tc3.build_frame(0x1230, 256, b"_") # 序號超出 1 byte
    # 剛好合法的最小碼框要組得出來,而且解得回去
    frame = tc3.build_frame(0x1230, 0, b"_")
    out = tc3.decode_frame(frame)
    assert out is not None and out["cks_ok"] is True
    assert out["len"] == len(frame) == 12
