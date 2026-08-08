#!/usr/bin/env bash
# 驗證 scripts/deploy_keep_runtime_config.sh 真的能在 git reset --hard 之後
# 保住現場的執行期設定檔。用真的 git repo 模擬一次部署，不是模擬語意。
#
# 回歸案例：2026-08-08 現場 config/frigate/config.yml 與 origin 差 83 行
# （cam_2 進出線 ROI 座標 + 已停用的 cam_3），下一次 deploy 的
# git reset --hard origin/main 就會把它洗掉，靠人工發現才救回來。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="bash $ROOT/scripts/deploy_keep_runtime_config.sh"
fails=0

check() {                     # check <名稱> <實際> <期望>
  if [ "$2" = "$3" ]; then
    echo "  PASS  $1"
  else
    echo "  FAIL  $1  got=[$2] want=[$3]"
    fails=$((fails + 1))
  fi
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# ── 建一個假的 origin + 工作區,重現 deploy 的形狀 ──────────────────
export GIT_CONFIG_GLOBAL="$tmp/gitconfig"    # 不要吃到使用者的 git 設定
git init -q --bare "$tmp/origin.git"
git clone -q "$tmp/origin.git" "$tmp/work"
cd "$tmp/work"
git config user.email t@t; git config user.name t

mkdir -p config/frigate config/system
echo "origin-config"   > config/frigate/config.yml
echo "origin-ui"       > config/frigate/ui_settings.json
echo "origin-feature"  > config/system/feature_state.json
echo "程式碼"          > app.py
git add -A && git commit -qm init && git push -q origin HEAD:main
git branch -q -M main
git branch -q --set-upstream-to=origin/main main 2>/dev/null

# origin 前進一版（模擬別人推了新 code）
echo "新版程式碼" > app.py
git add -A && git commit -qm feat && git push -q origin main

# 現場端：把 HEAD 退回舊版，並「在執行期」改設定檔
git reset -q --hard HEAD~1
echo "現場-ROI座標"  > config/frigate/config.yml     # 網頁 ROI 編輯器寫的
echo "現場-UI"       > config/frigate/ui_settings.json
#   feature_state.json 保持與 origin 相同 → 用來驗「相同就不該有動作」
echo "沒被列管的檔"  > config/frigate/other.txt

# ── 執行 deploy 流程 ──────────────────────────────────────────────
KEEP=$($SCRIPT save)
check "save 有回傳存在的暫存目錄" "$([ -d "$KEEP" ] && echo yes || echo no)" "yes"

git fetch -q origin main && git reset -q --hard origin/main

check "reset 後程式碼有更新到新版" "$(cat app.py)" "新版程式碼"
check "reset 確實會洗掉現場設定(這就是要修的問題)" "$(cat config/frigate/config.yml)" "origin-config"

restore_out=$($SCRIPT restore "$KEEP" 2>&1)

# ── 驗收 ──────────────────────────────────────────────────────────
check "現場改過的 config.yml 被保住"        "$(cat config/frigate/config.yml)"        "現場-ROI座標"
check "現場改過的 ui_settings.json 被保住"  "$(cat config/frigate/ui_settings.json)"  "現場-UI"
check "程式碼仍是 origin 新版(沒被連累)"    "$(cat app.py)"                           "新版程式碼"
check "與 origin 相同的檔案不動作"          "$(echo "$restore_out" | grep -c feature_state)" "0"
check "有列出實際保留的檔案"                "$(echo "$restore_out" | grep -c '保留現場設定')" "2"
check "暫存目錄用完清掉"                    "$([ -d "$KEEP" ] && echo yes || echo no)" "no"

# ── 邊界情形 ──────────────────────────────────────────────────────
KEEP2=$($SCRIPT save)
rm -f config/frigate/config.yml            # 現場沒有這檔(新站第一次部署)
$SCRIPT restore "$KEEP2" >/dev/null 2>&1
check "現場缺檔時會用保留的那份補回" "$(cat config/frigate/config.yml 2>/dev/null)" "現場-ROI座標"

$SCRIPT restore "/tmp/不存在的目錄-$$" >/dev/null 2>&1
check "暫存目錄不存在時不炸(回 0)" "$?" "0"

$SCRIPT >/dev/null 2>&1
check "沒給子命令要回非 0" "$?" "2"

echo
if [ "$fails" -ne 0 ]; then
  echo "FAIL $fails 項"
  exit 1
fi
echo "ALL PASS"
