#!/usr/bin/env python3
"""leader-follower 合批的正確性測試。

要證明四件事,缺任何一件就不能上線:
  ① 每一張拿回的是「自己的」結果,不會張冠李戴
  ② 併發時真的有合批(batch_size > 1)
  ③ 🛑 每個呼叫端仍然各自取得 GPU_INFERENCE_LOCK —— 競爭者數量不變。
     這正是中央批次器失敗的地方(N 條收斂成 1 條,對 LPR 的佔比就掉了),
     所以這條必須是硬性斷言,不是「應該有吧」。
  ④ 推論丟例外時呼叫端收到例外,而且其他人不會被卡住

用假模型:要驗的是分派與鎖的行為,不是 YOLO。
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

from detection.gpu_lock import GPU_INFERENCE_LOCK  # noqa: E402
from detection.leader_batch import LEADER, leader_batch_stats  # noqa: E402


class _FakeModel:
    def __init__(self, delay=0.03):
        self.delay = delay
        self.calls = []

    def __call__(self, frame, conf=None, verbose=None, device=None):
        batch = frame if isinstance(frame, list) else [frame]
        self.calls.append(len(batch))
        time.sleep(self.delay)
        return [f'r{f}' for f in batch]


def parse(result, frame):
    return [{'from': frame, 'result': result}]


def main() -> int:
    fail = []

    # 記錄鎖被取得幾次(gpu_lock 的統計就是照 acquire 計數的)
    before = leader_lock_samples()

    model = _FakeModel(delay=0.04)
    out = {}
    lk = threading.Lock()

    def worker(tag):
        r = LEADER.run(model, 0.15, 'cuda:0', tag, parse)
        with lk:
            out[tag] = r

    N = 6
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)

    # ① 結果對得上
    for i in range(N):
        if out.get(i) != [{'from': i, 'result': f'r{i}'}]:
            fail.append(f'第 {i} 張拿到 {out.get(i)}')

    # ② 有合批
    biggest = max(model.calls) if model.calls else 0
    print(f"  {N} 張併發:模型呼叫 {len(model.calls)} 次,批次大小 {model.calls},最大 {biggest}")
    if biggest < 2:
        fail.append(f'沒有合批(最大批次 {biggest})')

    # ③ 競爭者數量不變:N 個呼叫端 → 鎖至少被取得 N 次
    acquired = leader_lock_samples() - before
    print(f"  GPU 鎖取得次數 {acquired}(呼叫端 {N} 個)")
    if acquired < N:
        fail.append(f'鎖只被取得 {acquired} 次 < 呼叫端 {N} 個 —— 競爭者被收斂了,'
                    f'會重演中央批次器的公平性損失')

    # ④ 例外傳回,且不影響其他人
    class _Boom:
        def __call__(self, *a, **k):
            raise RuntimeError('模擬推論失敗')

    err_seen = []

    def boom_worker():
        try:
            LEADER.run(_Boom(), 0.15, 'cuda:0', 'x', parse)
        except RuntimeError as e:
            err_seen.append(str(e))

    ok_model = _FakeModel(delay=0.01)
    ok_out = []

    def ok_worker():
        ok_out.append(LEADER.run(ok_model, 0.15, 'cuda:0', 'y', parse))

    tb = threading.Thread(target=boom_worker)
    to = threading.Thread(target=ok_worker)
    tb.start(); to.start(); tb.join(timeout=10); to.join(timeout=10)
    if not err_seen or '模擬推論失敗' not in err_seen[0]:
        fail.append(f'推論失敗沒有正確傳回:{err_seen}')
    if ok_out != [[{'from': 'y', 'result': 'ry'}]]:
        fail.append(f'失敗的那筆影響到別人:{ok_out}')
    if not fail:
        print("  推論例外正確傳回,且不影響同時間的其他請求")

    st = leader_batch_stats()
    print(f"  統計:{ {k: st.get(k) for k in ('enabled','batch_size_avg','batch_size_max','samples')} }")
    print("  結果:", "❌ " + " / ".join(fail) if fail else "✅ 全過")
    return 1 if fail else 0


def leader_lock_samples() -> int:
    s = GPU_INFERENCE_LOCK.stats()
    return int(s.get('samples', 0))


if __name__ == "__main__":
    sys.exit(main())
