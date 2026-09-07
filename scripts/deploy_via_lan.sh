#!/usr/bin/env bash
# 現場區網直送部署 —— 不經 GitHub,給「Jetson 沒有對外網路」時用。
#
# 為什麼需要這支:
#   正常部署是 dev PC push → GitHub Actions → Jetson 上的 self-hosted runner
#   git pull + 重啟。但 runner 是**出向長輪詢**,Jetson 一旦沒有對外網路,
#   job 就永遠卡在 queued(2026-09-07 現場實際發生:gateway 不轉外網,
#   8.8.8.8/1.1.1.1/github 全不通,但區網完全正常)。
#   這支改走區網:把 commit 打成 git bundle → scp → 在 Jetson 上套用,
#   commit hash 與 GitHub 上**完全一致**,網路回來後 runner 補跑不會衝突。
#
# 用法:
#   bash scripts/deploy_via_lan.sh              # 部署到預設現場 IP
#   TRAFFIC_HOST=10.42.38.35 bash scripts/deploy_via_lan.sh
#   bash scripts/deploy_via_lan.sh --force      # 略過尖峰時段保護
#   bash scripts/deploy_via_lan.sh --dry-run    # 只看會做什麼,不動現場
#
# 🛑 這支會 git reset --hard,現場未提交的改動會消失 —— 與 GitHub Actions
#    的部署行為一致(jetson-verify.yml 也是 reset --hard)。現場的 runtime
#    設定由 scripts/deploy_keep_runtime_config.sh save/restore 保住。

set -euo pipefail

HOST="${TRAFFIC_HOST:-10.42.38.35}"
USER_AT="${TRAFFIC_USER:-ubuntu}"
REMOTE_DIR="${TRAFFIC_REMOTE_DIR:-/workspace}"
SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes)
FORCE=0
DRY=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "未知參數: $a"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[31m🛑 %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0) 尖峰時段保護 ────────────────────────────────────────────────
# 早上 07:00-12:00 是最重要的觀測時段(使用者定的),重啟會中斷抄錄與中央連線。
HH=$(date +%H)
if [ "$FORCE" -eq 0 ] && [ "$HH" -ge 7 ] && [ "$HH" -lt 12 ]; then
  die "現在 $(date +%H:%M) 在早尖峰觀測時段(07:00-12:00),部署會中斷抄錄與中央連線。
    真的要現在部署請加 --force。"
fi

# ── 1) 本機檢查 ───────────────────────────────────────────────────
say "本機檢查"
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then
  die "本機有未提交的改動,先 commit 再部署(避免送出去的東西跟 commit 對不起來):
$(git status --short | head -10)"
fi
LOCAL=$(git rev-parse HEAD)
info "本機 HEAD = $(git log --format='%h %s' -1)"

# ── 2) 連線與遠端狀態 ─────────────────────────────────────────────
say "現場連線"
ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" true 2>/dev/null \
  || die "SSH 連不到 $USER_AT@$HOST。確認在同一區網、或改 TRAFFIC_HOST。"
info "SSH $USER_AT@$HOST 可用"

REMOTE=$(ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" "cd $REMOTE_DIR && git rev-parse HEAD")
info "現場 HEAD = $(ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" "cd $REMOTE_DIR && git log --format='%h %s' -1")"

if [ "$LOCAL" = "$REMOTE" ]; then
  info "版本相同,不需要部署。"
  exit 0
fi
# 🛑 現場的 commit 本機必須看得到,否則做不出 bundle(也代表現場跑的是
#    本機沒有的東西 —— 那是更嚴重的問題,要先查清楚,不可以直接蓋掉)。
git cat-file -e "$REMOTE^{commit}" 2>/dev/null \
  || die "現場的 commit ${REMOTE:0:7} 在本機不存在。
    可能是現場被手動改過、或本機還沒 fetch。先查清楚再部署,不要直接覆蓋。"
if ! git merge-base --is-ancestor "$REMOTE" "$LOCAL"; then
  die "現場的 ${REMOTE:0:7} 不是本機 HEAD 的祖先(分岔了)。
    直接部署會讓現場的 commit 消失,先處理分岔。"
fi

say "將要部署的 commit"
git log --format='   %h %s' "$REMOTE..$LOCAL"
CHANGED=$(git diff --name-only "$REMOTE" "$LOCAL")
say "變動檔案"
echo "$CHANGED" | sed 's/^/   /'

# ── 3) 依變動決定要重啟哪些服務(規則與 jetson-verify.yml 一致)──────
# 🛑 traffic-signal / traffic-io 是獨立行程,只重啟 traffic-api 不會生效。
#    歷史教訓:2026-08-28 改的 signal_tc3.py 到 09-03 才被發現沒生效,
#    控制模式判讀空白了六天。這裡的比對規則必須與 workflow 同步。
RESTART_SIGNAL=0
RESTART_IO=0
echo "$CHANGED" | grep -qE '^(services/signal_daemon\.py|api/routes/signal_tc3\.py|detection/signal_timing_lookup\.py|config/tc3/|config/system/ramp_timing_baseline\.json)' && RESTART_SIGNAL=1
echo "$CHANGED" | grep -qE '^(services/io_(daemon|service)\.py|api/routes/(lock|io)_?.*\.py|api/utils/door_state\.py)' && RESTART_IO=1
say "將重啟的服務"
info "traffic-api      是(一律重啟)"
info "traffic-signal   $([ $RESTART_SIGNAL -eq 1 ] && echo '是(號誌抄錄相關檔案有變動)' || echo '否(無相關變動,避免中斷抄錄)')"
info "traffic-io       $([ $RESTART_IO -eq 1 ] && echo '是(IO/電子鎖相關檔案有變動)' || echo '否(無相關變動)')"

if [ "$DRY" -eq 1 ]; then
  printf '\n\033[33m--dry-run:到此為止,現場沒有被改動。\033[0m\n'
  exit 0
fi

# ── 4) 打 bundle 並送過去 ─────────────────────────────────────────
say "打包並傳送"
BUNDLE=$(mktemp -t deploy.XXXXXX.bundle)
trap 'rm -f "$BUNDLE"' EXIT
git bundle create "$BUNDLE" "$REMOTE..$LOCAL" >/dev/null 2>&1
info "bundle $(wc -c < "$BUNDLE") bytes"
scp "${SSH_OPTS[@]}" -q "$BUNDLE" "$USER_AT@$HOST:/tmp/deploy.bundle"
info "已送達 $HOST:/tmp/deploy.bundle"

# ── 5) 在現場套用 ─────────────────────────────────────────────────
say "現場套用"
ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" \
    "REMOTE_DIR='$REMOTE_DIR' RESTART_SIGNAL=$RESTART_SIGNAL RESTART_IO=$RESTART_IO bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$REMOTE_DIR"
BEFORE=$(git rev-parse --short HEAD)

# 保留現場 runtime 設定(Frigate 設定、feature_state、io_settings、nx_settings …)
KEEP=""
if [ -f scripts/deploy_keep_runtime_config.sh ]; then
  KEEP=$(bash scripts/deploy_keep_runtime_config.sh save)
fi

git fetch /tmp/deploy.bundle main:refs/remotes/origin/main -f
git reset --hard origin/main

if [ -n "$KEEP" ]; then
  bash scripts/deploy_keep_runtime_config.sh restore "$KEEP"
fi
AFTER=$(git rev-parse --short HEAD)
echo "   commit: $BEFORE → $AFTER"

echo "   重啟 traffic-api"
sudo -n systemctl restart traffic-api.service
if [ "$RESTART_SIGNAL" = "1" ]; then
  echo "   重啟 traffic-signal"
  sudo -n systemctl restart traffic-signal.service || echo "   ⚠ traffic-signal 重啟失敗,請人工確認"
fi
if [ "$RESTART_IO" = "1" ]; then
  echo "   重啟 traffic-io"
  sudo -n systemctl restart traffic-io.service || echo "   ⚠ traffic-io 重啟失敗,請人工確認"
fi
rm -f /tmp/deploy.bundle
REMOTE_SCRIPT

# ── 6) 驗證 ───────────────────────────────────────────────────────
say "驗證"
OK=0
for i in $(seq 1 20); do
  if ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" \
       "curl -sf -m 8 http://127.0.0.1:8000/api/health >/dev/null" 2>/dev/null; then
    OK=1; break
  fi
  sleep 3
done
[ "$OK" -eq 1 ] || die "服務起來後 60 秒內 /api/health 仍不通,請人工確認。"
info "健康檢查 通過"

DEPLOYED=$(ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" "cd $REMOTE_DIR && git rev-parse HEAD")
[ "$DEPLOYED" = "$LOCAL" ] || die "版本對不上:現場 ${DEPLOYED:0:7} ≠ 本機 ${LOCAL:0:7}"
info "版本一致 ${LOCAL:0:7}"

ssh "${SSH_OPTS[@]}" "$USER_AT@$HOST" '
  for s in traffic-api traffic-signal traffic-io; do
    printf "   %-16s %s\n" "$s" "$(systemctl is-active $s 2>/dev/null)"
  done
  n=$(ss -tn 2>/dev/null | grep -c ":1001" || true)
  printf "   %-16s %s 條(上游控制器 + 中央中繼)\n" "號誌 1001 連線" "$n"
'

printf '\n\033[32m✅ 部署完成 %s\033[0m\n' "${LOCAL:0:7}"
printf '   網路恢復後 GitHub Actions 會補跑同一個 commit,結果會是 no change。\n'
