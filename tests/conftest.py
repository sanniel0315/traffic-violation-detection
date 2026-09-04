"""把「腳本型」測試接進 pytest，讓 `pytest tests/` 能一次跑完整包。

## 為什麼需要這個檔

`tests/` 底下有兩種風格的測試，混在同一個資料夾：

1. **pytest 風格** —— 定義 `def test_xxx()`，pytest 直接收走。
2. **腳本風格** —— 從上到下跑完，自己數 fails，最後 `sys.exit(碼)`。
   設計上就是 `python tests/test_xxx.py` 直接執行，輸出是給人看的
   `PASS/FAIL` 清單。

第 2 種被 pytest 匯入時會在 **collection 階段**丟出 `SystemExit`，
pytest 收到後直接 `INTERNALERROR`，**整包測試當場停掉**（不是那一個檔失敗
而已，是後面所有檔都不跑了）。所以 `pytest tests/` 一直是壞的，
22 個腳本型測試從來沒有在 CI 跑過。

## 為什麼不是去改那 22 個腳本

改法會是「把整個 module body 縮排進函式」或「在每個檔塞一段 shim」。
前者是 22 個大 diff、每個都有改壞的風險；後者是同一段樣板複製 22 份。
兩種都在動「本來就是對的、而且平常有人在單獨執行」的測試。

改成在 collection 這一層處理：腳本型的檔案**開子行程跑、斷言結束碼為 0**。
腳本一行都不用改，兩種執行方式（`python tests/x.py` 與 `pytest tests/`）
都成立，而且結束碼本來就是這些腳本唯一的機器可讀輸出。

## 判斷規則

模組層級沒有任何 `test_*` 函式 → 當成腳本型。

這條規則同時修掉第二個問題：`test_vehicle_detector_equivalence.py` 之類
有 `if __name__ == "__main__": sys.exit(main())` 的檔案匯入不會爆，但也
**沒有 test 函式可收**，pytest 只會安靜地說「no tests ran」——
看起來是綠的，其實一個檢查都沒跑。現在它們會真的被執行。
"""
import ast
import os
import subprocess
import sys

import pytest

# 腳本型測試的逾時。載模型的那幾個(vehicle_detector_equivalence)要久一點。
SCRIPT_TIMEOUT_SEC = 600

# 🛑 在這裡就把環境快照起來,不能等到 runtest 才讀 os.environ。
#    pytest 會先 collection(把所有測試模組 import 一輪)再開始執行,
#    而不少測試模組在 import 時就寫 os.environ 來設定待測模組
#    (例:test_tc3_center_relay 設 SIGNAL_TC3_ENABLED=0)。
#    等到跑腳本時才抓,子行程會繼承到別支測試注入的變數 —— 實測就是
#    test_signal_tc3_recorder 單獨跑會過、整包跑被關掉抄錄器而失敗。
#    conftest 早於所有測試模組載入,這一刻的環境才是乾淨的。
_CLEAN_ENV = dict(os.environ)


def _has_test_functions(path) -> bool:
    """模組層級有沒有 pytest 收得到的 test 函式。

    用 AST 而不是 grep：`# def test_x` 這種註解不該算數，
    巢狀在 class/function 裡的也不是模組層級。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return True  # 讀不懂就交回給 pytest 自己報錯，不要在這裡吞掉
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return True
    return False


class ScriptFailure(Exception):
    """腳本結束碼非 0。訊息就是它自己印的 PASS/FAIL 清單。"""


class ScriptItem(pytest.Item):
    def runtest(self):
        env = dict(_CLEAN_ENV)
        # 腳本自己會 setdefault，這裡只是保證子行程一定有，
        # 免得某台機器的環境缺這個就整批紅掉。
        env.setdefault("AUTH_SECRET",
                       "test-only-secret-not-for-production-use-01234567")
        try:
            r = subprocess.run(
                [sys.executable, str(self.path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=SCRIPT_TIMEOUT_SEC,
                cwd=str(self.path.parent.parent), env=env,
            )
        except subprocess.TimeoutExpired:
            raise ScriptFailure(
                f"逾時（超過 {SCRIPT_TIMEOUT_SEC} 秒未結束）") from None
        if r.returncode != 0:
            raise ScriptFailure(
                f"結束碼 {r.returncode}\n\n--- stdout ---\n{r.stdout}"
                f"\n--- stderr ---\n{r.stderr}")

    def repr_failure(self, excinfo, style=None):
        # 腳本的輸出本身就是報告，原樣印出來比 pytest 的 traceback 有用
        if isinstance(excinfo.value, ScriptFailure):
            return f"{self.name} 腳本型測試失敗：{excinfo.value}"
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, 0, f"腳本型測試 {self.name}"


class ScriptFile(pytest.File):
    def collect(self):
        yield ScriptItem.from_parent(self, name=self.path.name)


# 🛑 這裡必須用 pytest_pycollect_makemodule,不能用 pytest_collect_file。
#    pytest_collect_file 的結果是「各家外掛的收集結果並存」—— 我方回傳
#    ScriptFile 之後,內建 python 外掛照樣會再建一個 Module 去匯入同一個檔,
#    SystemExit 一樣會炸。makemodule 是 firstresult 型:先回傳的人直接取代
#    模組節點,內建外掛不會再匯入。
@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name == "conftest.py" or _has_test_functions(module_path):
        return None  # 正常 pytest 檔,交給預設流程
    return ScriptFile.from_parent(parent, path=module_path)
