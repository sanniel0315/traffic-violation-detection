"""號控的安全閘門。

這些檢查是唯一擋在「按錯一個按鈕就對運轉中的號誌送出位元組」前面的東西,
所以每一條都要有測試。壞掉的話不會有錯誤訊息 —— 只會有一個路口被改掉。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def tc3():
    import importlib.util
    os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret-" + "x" * 24)
    spec = importlib.util.spec_from_file_location(
        "_tc3_ctl", ROOT / "api" / "routes" / "signal_tc3.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"signal_tc3 匯入失敗: {exc}")
    return mod


def test_型態以規範目錄為準(tc3):
    """分類錯了,擋不擋得住就全錯。

    🛑 不能只看指令碼高位元組 —— 用 105 條目錄比對過,高位元組 8 那格是混的:
       0F8E「密碼代碼－設定」和 0F8F「設定回報」都落在 8。
       所以 _kind_of 以目錄的 message_type 為準,目錄查不到才退回位元組規則。
    """
    assert tc3._kind_of(0x03, 0x5F) == "主動回報"   # 5F03 燈態主動回報
    assert tc3._kind_of(0x15, 0x5F) == "設定"       # 5F15 時制計畫
    assert tc3._kind_of(0x45, 0x5F) == "查詢"       # 5F45 時制計畫查詢
    assert tc3._kind_of(0xC0, 0x0F) == "查詢回報"
    # 就是那個例外:同樣是高位元組 8,一個是設定一個是回報
    assert tc3._kind_of(0x8E, 0x0F) == "設定"
    assert tc3._kind_of(0x8F, 0x0F) == "設定回報"


def test_位元組規則本身也要對(tc3):
    """目錄查不到的碼要退回這個規則,所以它也要正確。"""
    assert tc3._kind_by_nibble(0x03) == "主動回報"
    assert tc3._kind_by_nibble(0x15) == "設定"
    assert tc3._kind_by_nibble(0x45) == "查詢"
    assert tc3._kind_by_nibble(0x80) == "設定回報"
    assert tc3._kind_by_nibble(0xC0) == "查詢回報"


def test_總開關關著時什麼都不准送(tc3, monkeypatch):
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", False)
    for cmd in (0x45, 0x15, 0x10):
        why = tc3._control_guard(cmd)
        assert why, f"總開關關著卻放行了 cmd={cmd:02X}"
        assert "未啟用" in why


def test_控制器回給中心的訊息中心不會送(tc3, monkeypatch):
    """主動回報/設定回報/查詢回報都是控制器→中心,中心送這些沒有意義。"""
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", False)
    for cmd, dev in ((0x03, 0x5F), (0x8F, 0x0F), (0xC0, 0x0F)):
        why = tc3._control_guard(cmd, dev)
        assert why, f"{dev:02X}{cmd:02X} 應該被擋"


def test_只准查詢時設定類要被擋(tc3, monkeypatch):
    """這是預設值,也是驗證 TX 通不通時唯一該開的狀態。"""
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", True)
    assert tc3._control_guard(0x45) is None, "查詢應該放行"
    for cmd in (0x15, 0x10, 0x18, 0x1C):        # 時制/控制策略/指定時制/步階變換
        why = tc3._control_guard(cmd)
        assert why, f"只准查詢卻放行了 cmd={cmd:02X}"
        assert "只准查詢" in why


def test_開放設定類之後才放行(tc3, monkeypatch):
    """🛑 開放設定類**還不夠** —— 還要動態控制總開關是開的。

    規範 (E) 要求遠端開關、(D) 要求降階運轉,兩者都落在把關層而不是介面層:
    只擋按鈕的話,知道 API 的人照樣送得出去。
    """
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", False)
    old = dict(tc3._dyn)
    try:
        # 總開關關著 → 設定類仍被擋,但**查詢類照樣放行**
        tc3._dyn.update({"enabled": False, "level": "L0", "reason": ""})
        for cmd in (0x15, 0x10, 0x1C):
            why = tc3._control_guard(cmd)
            assert why and "總開關" in why, f"總開關關著卻放行了 cmd={cmd:02X}"
        assert tc3._control_guard(0x45) is None, "查詢類不該被總開關擋 —— 它不改變運轉"

        # 總開關開了 → 放行
        tc3._dyn.update({"enabled": True})
        for cmd in (0x15, 0x10, 0x45):
            assert tc3._control_guard(cmd) is None

        # 降階 L2 → 設定類再度被擋,查詢仍可用(降階時更需要查現場狀況)
        tc3.enter_degraded("L2", "單元測試")
        for cmd in (0x15, 0x1C):
            why = tc3._control_guard(cmd)
            assert why and "降階" in why, f"降階中卻放行了 cmd={cmd:02X}"
        assert tc3._control_guard(0x45) is None, "降階時查詢也被擋了 —— 那會看不到現場"
    finally:
        tc3._dyn.update(old)


def test_送出用的碼框組得出來且解得回去(tc3):
    """5F45 = 時制計畫查詢,參數帶 PlanID。"""
    frame = tc3.build_frame(0x1230, 5, bytes([0x5F, 0x45, 0x05]))
    out = tc3.decode_frame(frame)
    assert out is not None and out["cks_ok"] is True
    assert out["code"] == "5F45"
    assert out["addr"] == 0x1230
    assert out["len"] == len(frame)


def test_位址推得規則(tc3, monkeypatch):
    """猜錯位址等於把命令送給別的路口,所以來源要明確。"""
    monkeypatch.setattr(tc3, "CONTROL_ADDR", 0x9999)
    assert tc3._target_addr() == 0x9999          # 有設定就用設定

    monkeypatch.setattr(tc3, "CONTROL_ADDR", 0)
    tc3._frames.clear()
    assert tc3._target_addr() is None            # 沒設定也沒抄到 → 不要亂猜
    tc3._frames.append({"addr": 0x1230})
    assert tc3._target_addr() == 0x1230          # 用抄到的


def test_reassert只在持有控制權時作用(tc3, monkeypatch):
    """🛑 reassert 是繞過中央自動保護,但不可以無條件生效。

    只有在「總開關開 + 未降階」時才重新宣告 —— 平時中央的策略設定照過,
    否則等於把路口控制權從中央手上搶走,那是權責問題不是技術選項。
    """
    calls = []
    monkeypatch.setitem(tc3._downlink_policy, "v", "reassert")
    monkeypatch.setattr(tc3.threading, "Timer",
                        lambda delay, fn: type("T", (), {"start": lambda s: calls.append(delay)})())
    rec = {"code": "5F10", "raw": "AA BB 21 FF FF 00 0E 5F 10 01 00 AA CC 16"}   # 中央送 0x01
    old = dict(tc3._dyn)
    try:
        # 沒持有控制權 → 不重新宣告
        tc3._dyn.update({"enabled": False, "level": "L0"})
        tc3._downlink_allow(dict(rec))
        assert not calls, "沒持有控制權卻重新宣告了"

        # 持有控制權 + 中央收走 bit4 → 重新宣告
        tc3._dyn.update({"enabled": True, "level": "L0"})
        tc3._downlink_allow(dict(rec))
        assert calls, "持有控制權卻沒有重新宣告"

        # 降階中 → 不重新宣告(降階的動作就是什麼都不做)
        calls.clear()
        tc3._dyn.update({"enabled": True, "level": "L2"})
        tc3._downlink_allow(dict(rec))
        assert not calls, "降階中卻重新宣告了"

        # 中央送的策略已含 bit4 → 沒在收走權限,不必重新宣告
        calls.clear()
        tc3._dyn.update({"enabled": True, "level": "L0"})
        tc3._downlink_allow({"code": "5F10",
                             "raw": "AA BB 21 FF FF 00 0E 5F 10 10 01 AA CC 16"})
        assert not calls, "中央沒有收走 bit4,不該重新宣告"
    finally:
        tc3._dyn.update(old)


def test_確認手動時自動降階L2(tc3, monkeypatch):
    """🛑 規範 (G):員警手動時我方要完全讓開。

    掛在「已確認」的手動上,不是原始位元 —— 5F10 續約瞬間策略會閃過 05H,
    用原始位元判會每小時假降階十次(2026-09-03 教訓)。
    """
    # 🛑 要驗的是「降階會擋」,所以前面兩道把關必須先放行,否則擋下來的是
    #    CONTROL_ENABLED 而不是降階 —— 測試會過但驗錯了東西。
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", False)
    old_dyn, old_safety = dict(tc3._dyn), dict(tc3._safety)
    try:
        tc3._dyn.update({"enabled": True, "level": "L0", "reason": ""})
        # 確認手動 → L2
        tc3.enter_degraded("L2", "偵測到手動介入(定時控制+路口手動),我方停止下發")
        assert tc3._dyn["level"] == "L2"
        assert tc3.dynamic_blocked(), "降階後仍放行設定類"
        assert tc3._control_guard(0x1C) and "降階" in tc3._control_guard(0x1C)
        # 查詢類在降階時仍要能用 —— 降階時更需要看現場
        assert tc3._control_guard(0x45) is None
        # 手動解除 → 回 L0
        tc3.enter_degraded("L0", "手動已解除")
        assert tc3._dyn["level"] == "L0"
        assert tc3.dynamic_blocked() is None
    finally:
        tc3._dyn.update(old_dyn)
        tc3._safety.update(old_safety)


def test_續約策略必須包含路口手動位元(tc3):
    """🛑 2026-09-08 現場:「切不了路口手動」。

    ControlStrategy 是「允許哪些控制來源」的遮罩。我方每 45 秒送一次
    5F10 續約,若只寫 bit4(0x10),等於每 45 秒把「允許路口手動」關掉一次 ——
    現場在控制箱切手動,最多撐 45 秒就被清掉,操作員會以為手動壞了。
    現場必須永遠切得動手動,所以續約值一定要含 bit2。
    """
    assert tc3.REASSERT_STRATEGY & tc3._BIT_PHASE, "續約必須保有時相控制 bit4"
    assert tc3.REASSERT_STRATEGY & tc3._BIT_ROADSIDE, \
        "續約必須一併允許路口手動 bit2,否則現場切不了手動"


def test_現場已切手動時不再續約(tc3, monkeypatch):
    """降階要 8 秒確認,續約每 45 秒一次 —— 若續約落在那 8 秒內,
    操作員的手動會在被確認之前就被我方寫回去。這一道把競態關掉。
    """
    sent = []
    monkeypatch.setattr(tc3, "_controller_send", lambda f: sent.append(f) or True)
    monkeypatch.setattr(tc3, "_target_addr", lambda: 0x0001)
    monkeypatch.setitem(tc3._dyn, "enabled", True)
    monkeypatch.setitem(tc3._dyn, "level", "L0")

    # 真手動 = 路側手動且**沒有**定時控制 → 不可以再送 5F10
    monkeypatch.setitem(tc3._safety, "strategy", tc3._BIT_ROADSIDE)
    tc3._do_reassert(kind="續約")
    assert sent == [], "現場手動中卻仍送出續約,會把操作員的手動蓋掉"
    assert "手動" in tc3._reassert["last_error"]

    # 我方仍持有 bit4(時相控制中)→ 照常續約
    monkeypatch.setitem(tc3._safety, "strategy", tc3._BIT_PHASE)
    tc3._do_reassert(kind="續約")
    assert len(sent) == 1, "正常情況下應該要續約"


def test_動態控制總開關要持久化(tc3, tmp_path, monkeypatch):
    """🛑 2026-09-08:為了修機箱門上傳而重啟 traffic-signal,總開關是純記憶體
    狀態,重啟後靜靜回到關閉 —— 路口退回固定時制、演算法停擺 45 分鐘,
    畫面上沒有任何地方顯示那是重啟造成的。

    存檔要把開關寫進去,讀檔要讀回來,重啟才會回到操作者最後設定的狀態。
    """
    cfg = tmp_path / "conn.json"
    monkeypatch.setattr(tc3, "_CONN_PATH", str(cfg))

    monkeypatch.setitem(tc3._conn, "dynamic_control", True)
    tc3._save_conn_config()
    assert '"dynamic_control": true' in cfg.read_text(encoding="utf-8")

    # 讀回來:換成 False 再載入,應該被檔案裡的 True 蓋回去
    monkeypatch.setitem(tc3._conn, "dynamic_control", False)
    tc3._load_conn_config()
    assert tc3._conn["dynamic_control"] is True, "重啟後沒有回到操作者最後設定的狀態"

    monkeypatch.setitem(tc3._conn, "dynamic_control", False)
    tc3._save_conn_config()
    monkeypatch.setitem(tc3._conn, "dynamic_control", True)
    tc3._load_conn_config()
    assert tc3._conn["dynamic_control"] is False, "關閉狀態也要能被持久化"


def test_授權到期殘留的手動位元不可以擋住續約(tc3, monkeypatch):
    """🛑 2026-09-08 實際把路口鎖死 35 分鐘的 bug,不可以再犯。

    我方續約值是 0x14(含 bit2「允許路口手動」)。5F10 授權到期後控制器
    **不會清掉 roadSideManual 位元**,會殘留為 1,策略變成
    0x05 = 定時控制 + 殘留 bit2。若用 raw 位元判「有手動位元就不續約」,
    等於在保護我方自己寫進去的位元 —— 續約永遠不會再送出,現場在系統上
    怎麼切演算法下發都沒反應,而且看不出原因。

    真手動的判定是「路側手動且**沒有**定時控制」(_control_mode 早就這樣寫)。
    """
    sent = []
    monkeypatch.setattr(tc3, "_controller_send", lambda f: sent.append(f) or True)
    monkeypatch.setattr(tc3, "_target_addr", lambda: 0x0001)
    monkeypatch.setitem(tc3._dyn, "enabled", True)
    monkeypatch.setitem(tc3._dyn, "level", "L0")

    # 0x05 = 定時控制 + 授權到期殘留的路側手動位元 → 必須照常續約
    monkeypatch.setitem(tc3._safety, "strategy",
                        tc3._BIT_FIXTIME | tc3._BIT_ROADSIDE)
    tc3._do_reassert(kind="續約")
    assert len(sent) == 1, ("授權到期殘留的 bit2 擋住了續約 —— "
                            "路口會永遠回不到動態控制")


def test_hwstatus_每個模式送出的值(tc3):
    """🛑 2026-09-08 這裡出過一個真的上線路的 bug。

    新增 swap 時只在最後補了交換那一步,沒給它自己的取值分支 ——
    swap 掉進 else(flip14),先把 bit14 XOR 掉才交換,
    送出 0x0022 而不是 0x0062,中央看到的「控制器就緒」整個不見了。
    當時的單元測試只驗了交換算術,沒驗**模式分派**,所以沒抓到;
    是現場 dry run 才發現的(錯誤值在線路上約 20 秒)。
    這一則逐模式驗實際送出的值。
    """
    f = tc3._hw_for_center
    RAW = 0x6000          # 現場常態值:bit13 外部時相控制中 + bit14 控制器就緒

    assert f(RAW, "raw") == 0x6000, "raw 必須原封不動"
    assert f(RAW, "swap") == 0x0060, "swap 掉了位元 —— 就是當初那個 bug"
    assert f(RAW, "zero") == 0x0000
    assert f(RAW, "force", 0x1234) == 0x1234
    assert f(RAW, "flip14") == 0x2000, "flip14 只翻 bit14"

    # 機箱門開啟:先在我方位元語意下設 bit9,最後一步才交換
    assert f(RAW, "raw", cabinet_open=True) == 0x6200
    assert f(RAW, "swap", cabinet_open=True) == 0x0062, (
        "順序錯了 —— 先交換再設機箱位元會設到錯的位置")

    # swap 必須可逆:交換兩次回到原值,一個位元都不能掉
    for v in (0x6000, 0x6200, 0x4000, 0x4004, 0x0001, 0xFFFF, 0x0000, 0x1234):
        once = f(v, "swap")
        assert f(once, "swap") == v, "0x%04X 交換兩次沒回到原值" % v


def test_hwstatus_模式白名單含swap(tc3):
    """模式清單漏掉 swap 的話,畫面切了會 400,而且看不出為什麼。"""
    import inspect
    src = inspect.getsource(tc3.control_hwstatus_mode)
    assert '"swap"' in src, "control_hwstatus_mode 沒有放行 swap"
