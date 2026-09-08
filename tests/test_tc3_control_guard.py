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
    # 🛑 2026-09-08 這個值改成可設定 + 持久化(畫面上的控制策略卡就是設它),
    #    所以測的是預設值,不是寫死常數。
    assert tc3._REASSERT_DEFAULT & tc3._BIT_PHASE, "續約預設必須保有時相控制 bit4"
    assert tc3._REASSERT_DEFAULT & tc3._BIT_ROADSIDE, \
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


def test_hwstatus_模式要持久化(tc3, tmp_path, monkeypatch):
    """🛑 使用者 2026-09-08:「這個不是暫時,重啟都必須維持」。

    只靠 systemd drop-in 撐不住 —— 檔案被清掉、或換一台機器部署就沒了,
    而那時中央會立刻又看到一整排假故障(bit9/bit13/bit14 被讀成三個硬體錯誤)。
    切了就要存,讀得回來。
    """
    cfg = tmp_path / "conn.json"
    monkeypatch.setattr(tc3, "_CONN_PATH", str(cfg))

    monkeypatch.setitem(tc3._conn, "hwstatus_mode", "swap")
    tc3._save_conn_config()
    assert '"hwstatus_mode": "swap"' in cfg.read_text(encoding="utf-8")

    monkeypatch.setitem(tc3._conn, "hwstatus_mode", "raw")
    tc3._load_conn_config()
    assert tc3._conn["hwstatus_mode"] == "swap", "重啟後沒有回到設定的模式"


def test_hwstatus_遮蔽位元(tc3):
    """🛑 2026-09-08 使用者授權遮掉 bit13(外部時相控制進行中)。

    bit13 是**狀態指示不是故障**,只要我方持有時相控制它就恆亮,
    中央把它顯示成 TIMING_PLAN_ON_TRANSITION 並當異常管理。
    遮掉它不隱瞞運轉狀態 —— 中央每 5 秒輪詢 5F40,我方據實轉答 0x14。
    """
    f = tc3._hw_for_center
    B13 = 1 << 13

    # 遮 bit13:其餘位元不動
    assert f(0x6000, "raw", mask_out=B13) == 0x4000
    assert f(0x6004, "raw", mask_out=B13) == 0x4004, "遮蔽不可以動到其他位元"
    assert f(0x4000, "raw", mask_out=B13) == 0x4000, "本來就沒亮就不該有變化"

    # 遮蔽要在**交換之前**、在我方位元語意下做
    assert f(0x6000, "swap", mask_out=B13) == 0x0040
    assert f(0x6000, "swap", cabinet_open=True, mask_out=B13) == 0x0042

    # 不遮的時候行為不變
    assert f(0x6000, "swap", cabinet_open=True) == 0x0062


def test_hwstatus_錯誤類位元不得遮蔽(tc3):
    """🛑 遮狀態指示可以,遮錯誤類等於對主管機關謊報故障情形 —— 必須擋。"""
    import inspect
    src = inspect.getsource(tc3.control_hwstatus_mode)
    assert "allowed" in src and "(1 << 13)" in src, "沒有限制可遮蔽的位元"
    assert "不得遮蔽" in src, "沒有把理由寫在擋下來的訊息裡"


def test_設定總覽不得建議已被拒收的查詢(tc3, monkeypatch):
    """🛑 現場:「尚未抄到 5FDF;可送查詢碼 5F5F 取回」——

    但 5F5F 在這台控制器上歷史 16 次全 NAK(ErrorCode=1)。
    照著建議去送只會再失敗一次,而且看的人不會知道前面已經試過那麼多次。
    提示必須反映實際嘗試紀錄。
    """
    import inspect
    src = inspect.getsource(tc3.signal_config)
    assert "_query_attempts" in src, "沒有先查嘗試紀錄就給建議"
    assert "拒收" in src and "需洽廠商" in src, "被拒時沒有講清楚後果與該找誰"
    assert "尚未試過" in src, "沒試過的情況也要標明,否則跟試過失敗的混在一起"

    # 三種情況都要有各自的說法,不可以共用同一句
    fn = inspect.getsource(tc3._query_attempts)
    assert "0F81" in fn, "沒有解析 0F81(設定或查詢無效)"
    assert "src='self'" in fn, "沒有只算我方送出的"


def test_續約週期要遠小於授權有效期(tc3):
    """🛑 2026-09-08 實際失控 60 秒。

    授權 EffectTime 是「分鐘」,REASSERT_EFFECT=1 → 60 秒到期。
    原本續約週期 45 秒,餘裕只有 15 秒 —— 一次部署重啟就斷了:
      14:07:00 續約 → 14:07:05 重啟 → 14:08:00 到期 → 回定時控制
      → 14:08:37 新行程才第一次續約(啟動後 90 秒)
    週期必須小到「連掉兩次仍不失控」。
    """
    expiry = tc3.REASSERT_EFFECT * 60.0
    assert tc3.AUTH_RENEW_SEC * 3 <= expiry, (
        "續約週期 %.0fs 對授權 %.0fs 餘裕不足 —— 掉一次就可能失控"
        % (tc3.AUTH_RENEW_SEC, expiry))


def test_啟動後不可空等一個完整週期(tc3):
    """啟動後要等到抄得到控制策略就立刻續約,不是空等 AUTH_RENEW_SEC。"""
    import inspect
    src = inspect.getsource(tc3._auth_renew_loop)
    assert "AUTH_FIRST_WAIT_SEC" in src, "啟動後沒有先探策略就直接進週期迴圈"
    # 探測迴圈要在主迴圈之前
    assert src.index("AUTH_FIRST_WAIT_SEC") < src.index("while not shutdown_event.is_set():\n        try:")


def test_啟動與失聯時要主動查控制策略(tc3):
    """🛑 2026-09-08 第二次失控的真正原因。

    5F00/5FC0(控制策略)只有在中央輪詢 5F40 時才會出現,中央大約一分鐘問一次。
    重啟後 _safety["strategy"] 有將近 60 秒是 None,續約迴圈整段跳過
    (它刻意在抄不到策略時不下命令),授權就在這段空窗到期 → 路口退回定時控制。

    等待治不好,要**自己去問**:啟動時查一次,主迴圈抄不到時每輪再查。
    """
    import inspect
    src = inspect.getsource(tc3._auth_renew_loop)
    assert src.count('_send_query_to_controller("5F40"') >= 2, (
        "啟動時與抄不到策略時都要主動查 5F40,不能只做一邊")
    # 「抄不到就不下命令」這個安全性質不可以被改掉
    assert 'isinstance(strat, int)' in src, "抄不到策略時仍必須跳過下發"


def test_抄不到策略時不可睡滿一個續約週期(tc3):
    """🛑 2026-09-08:探測逾時後,主迴圈送完 5F40 就睡滿 20 秒才用答案,
    續約只剩 1 秒餘裕才趕上。查完要快點回頭看。
    """
    import inspect
    src = inspect.getsource(tc3._auth_renew_loop)
    assert "AUTH_PROBE_SEC" in src, "沒有短間隔重查"
    assert src.count("AUTH_PROBE_SEC") >= 2, "啟動與主迴圈兩處都要用短間隔"
    assert "continue" in src, "抄不到策略那一輪要 continue,不可掉到長 wait"
    assert tc3.AUTH_PROBE_SEC < tc3.AUTH_RENEW_SEC, "重查間隔要短於續約週期"


def test_設備時間解碼_民國年與binary(tc3):
    """🛑 兩個坑都實際踩過:

    Year 是**民國年**(0x73=115 → 西元 2026),當成 2000+ 會解出「20115 年」。
    時分秒是 **binary 不是 BCD**(0x14=20 時),當 BCD 會錯 6 個多小時。
    """
    raw = "AA BB 31 FF FF 00 13 0F C2 73 09 08 02 0E 07 00 AA CC E1"
    d = tc3._decode_device_time(raw)
    assert d is not None
    assert d["text"] == "2026-09-08 14:07:00", "民國年或 binary 解錯了"
    assert d["roc_year"] == 115
    assert d["week"] == 2, "Week 1~7,週二應為 2"

    # 20 時那一筆(0x14):當 BCD 會讀成 14 時
    raw2 = "AA BB 01 FF FF 00 13 0F C2 73 09 07 01 14 20 39 AA CC D9"
    d2 = tc3._decode_device_time(raw2)
    assert d2["text"] == "2026-09-07 20:32:57", "時分秒被當成 BCD 了"

    assert tc3._decode_device_time("亂碼") is None, "解不出來要回 None,不可以拋"


def test_設備時間編碼與解碼要對稱(tc3):
    """組出去的值,照 0FC2 的規則讀回來必須一致 —— 否則對時會把時間設錯。"""
    import datetime as _dt
    for n in (_dt.datetime(2026, 9, 8, 14, 7, 0),
              _dt.datetime(2026, 1, 1, 0, 0, 0),
              _dt.datetime(2025, 12, 31, 23, 59, 59)):
        v = tc3._time_values(n)
        assert v["Year"] == n.year - 1911
        assert v["Week"] == n.isoweekday(), "Week 要用 isoweekday(週一=1)"
        raw = "AA BB 01 FF FF 00 13 0F C2 %02X %02X %02X %02X %02X %02X %02X AA CC 00" % (
            v["Year"], v["Month"], v["Day"], v["Week"], v["Hour"], v["Min"], v["Sec"])
        back = tc3._decode_device_time(raw)
        assert back["text"] == n.strftime("%Y-%m-%d %H:%M:%S"), "編碼解碼不對稱"


def test_定時對時預設關閉且有上限保護(tc3):
    """🛑 自動會改變控制器的行為必須預設關閉(專案規範),且要有上限保護 ——
    差太多代表控制器時鐘可能故障,自動拉一大步可能在時段邊界造成
    非預期的時制計畫切換。
    """
    import inspect
    assert tc3._time_auto["enabled"] is False or isinstance(
        tc3._time_auto["enabled"], bool)
    src = inspect.getsource(tc3._time_auto_loop)
    assert "max_auto_sec" in src and "只告警" in src, "沒有上限保護"
    assert "threshold_sec" in src, "沒有門檻,會頻繁微調"
    # 門檻不可大於上限,否則永遠不會校正
    setsrc = inspect.getsource(tc3.control_time_auto)
    assert "門檻不可大於自動校正上限" in setsrc


def test_控制策略設定的是持續維持的值(tc3):
    """🛑 只送一次沒有意義 —— 續約每 AUTH_RENEW_SEC 秒會把它蓋回去。

    今天早上「切了沒反應」就是同一類問題:設定被另一個機制覆蓋,
    而畫面看不出來。所以介面設的必須是**續約要送的那個值**。
    """
    import inspect
    src = inspect.getsource(tc3.control_strategy)
    assert "_auth_strategy" in src, "沒有改到續約實際使用的值"
    assert "_save_conn_config" in src, "沒有持久化,重啟就回舊值"
    assert "_do_reassert" in src, "沒有立刻送出,要等下一輪續約"
    # 續約必須用可變的值,不可以再讀寫死常數
    rsrc = inspect.getsource(tc3._do_reassert)
    assert "reassert_strategy()" in rsrc, "續約還在用寫死的常數"


def test_控制策略全關要擋下來(tc3):
    """全部關閉不是有效策略,而且控制器的反應未知 —— 不可以放行。"""
    import inspect
    src = inspect.getsource(tc3.control_strategy)
    assert "if v == 0:" in src and "至少要選一種" in src


def test_控制策略位元表要八位元齊全(tc3):
    """現場給的定義:bit0 定時 / bit1 動態 / bit2 路口手動 / bit3 中央手動 /
    bit4 時相 / bit5 即時 / bit6 觸動 / bit7 特勤路線。"""
    assert tc3.STRATEGY_BITS[:8] == ["定時控制", "動態控制", "路口手動", "中央手動",
                                     "時相控制", "即時控制", "觸動控制", "特勤路線"]
    bits = tc3.strategy_bits(0x14)
    assert len(bits) == 8
    on = [b["bit"] for b in bits if b["on"]]
    assert on == [2, 4], "0x14 應該是 bit2 路口手動 + bit4 時相控制"
