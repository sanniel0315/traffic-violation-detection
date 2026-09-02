"""對外 API 的 null 語意:「沒有量到」才是 null,量到 0 就要回 0。

對外約定(即時車流查詢說明 §四.3)白紙黑字:
    null 代表「沒有量到」,不是 0
    avg_speed_kmh: null → 未設定測速 / avg_speed_kmh: 0 → 真的量到 0
    這兩者意義完全不同,請勿把 null 當成 0 處理。排隊長度等欄位同理。

🛑 歷史 bug(2026-09-03):原本寫 `round(v or 0,1) if v else None`,
   0 是 falsy → 「量到 0 公尺」被回成 null。後果:OPAC 每 5 秒拉
   /realtime,今日 4075 筆中四台排隊全 null 佔 57.6%(各台 76~83%),
   它把 null 當 swl=0,綠側空/紅側積 16m 時判兩側平手 → 綠燈一直亮。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.external import _num_or_none


def test_zero_stays_zero_not_null():
    """★核心:量到 0 就回 0.0,不可變成 None。"""
    assert _num_or_none(0) == 0.0
    assert _num_or_none(0.0) == 0.0
    assert _num_or_none("0") == 0.0


def test_none_stays_none():
    """沒量到才回 None。"""
    assert _num_or_none(None) is None


def test_normal_values_rounded():
    assert _num_or_none(16.04) == 16.0
    assert _num_or_none(37.5) == 37.5
    # 註:Python round() 是銀行家捨入,37.55→37.5,此處不驗邊界


def test_zero_is_not_none_explicitly():
    """0 與 None 必須可區分 —— 這正是對外文件強調的『意義完全不同』。"""
    assert _num_or_none(0) is not None


def test_no_falsy_check_left_in_queue_fields():
    """回歸防護:排隊/時長欄位不可再出現 falsy 判斷的寫法。

    掃原始碼,確保沒有 `if xxx.get("...QueueLengthM") else None` 這種
    會把 0 判成 None 的寫法復活。
    """
    src = (ROOT / "api" / "routes" / "external.py").read_text(encoding="utf-8")
    for key in ("avgQueueLengthM", "maxQueueLengthM",
                "queueDurationSec", "maxQueueDurationSec"):
        bad = 'if %s' % key
        # 允許出現在 _num_or_none(...) 的參數裡,但不可出現在三元判斷的條件位置
        for line in src.splitlines():
            if key in line and " else None" in line and "_num_or_none" not in line:
                raise AssertionError(
                    "排隊欄位又出現 falsy 判斷(0 會被當成 None):%s" % line.strip())


def test_no_or_none_on_queue_assignments():
    """回歸防護:排隊/時長的賦值不可再出現 `or None`。

    2026-09-03 第二處 bug:SQL 聚合後寫成
        if r[3] is not None:
            target["avgQueueLengthM"] = round(float(r[3]), 1) or None
    前面已判斷過有無資料,後面的 `or None` 會把「四捨五入後 0.0」的合法值
    變成 None。實測 cam4 有 81 筆樣本、零筆 null、avg=0.14,仍回 null。
    """
    src = (ROOT / "api" / "routes" / "external.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("#"):
            continue
        if ("QueueLengthM" in t or "QueueDurationSec" in t) and "or None" in t:
            raise AssertionError("排隊欄位賦值不可用 `or None`:%s" % t)
