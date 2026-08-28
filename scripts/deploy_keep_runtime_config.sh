#!/usr/bin/env bash
# 部署時保留「執行期設定檔」—— 讓 git reset --hard 不會洗掉現場設定。
#
# 背景：config/frigate/config.yml 這類檔案是程式在跑的時候自己寫的
# （網頁 ROI 編輯器、功能開關、IO / NVR 設定），不是原始碼。
# 它們同時 git-tracked，而 deploy 會 git reset --hard origin/main
# → 每次部署都把現場設定洗回 origin 版。
# 實測 2026-08-08 差 83 行（cam_2 進出線 ROI 座標 + 已停用的 cam_3），
# 那次是人工發現才救回來的；DEPLOY_NEW_SITE.md 早就把它列為「必解的雷」。
#
# 規則：repo 裡的版本從此只當「新站種子」，現場跑的永遠贏。
# 要從 git 派設定到現場，得先在現場刪掉該檔再部署。
#
# 用法（deploy 腳本內）：
#   KEEP=$(scripts/deploy_keep_runtime_config.sh save)
#   git reset --hard origin/main
#   scripts/deploy_keep_runtime_config.sh restore "$KEEP"
#
# save 會把暫存目錄路徑印到 stdout（其餘訊息一律走 stderr，才不會污染它）。
set -euo pipefail

# 要保留的檔案清單。新增「程式會自己寫、又被 git 追蹤」的設定檔時加在這裡。
RUNTIME_CONFIGS="${RUNTIME_CONFIGS:-\
config/frigate/config.yml \
config/frigate/ui_settings.json \
config/frigate/go2rtc.yaml \
config/system/feature_state.json \
config/system/io_settings.json \
config/system/nx_settings.json \
config/system/signal_conn.json}"

cmd="${1:-}"

case "$cmd" in
  save)
    keep=$(mktemp -d)
    for f in $RUNTIME_CONFIGS; do
      if [ -f "$f" ]; then
        mkdir -p "$keep/$(dirname "$f")"
        cp -a "$f" "$keep/$f"
      fi
    done
    echo "$keep"
    ;;

  restore)
    keep="${2:-}"
    if [ -z "$keep" ] || [ ! -d "$keep" ]; then
      echo "restore: 暫存目錄不存在，略過（$keep）" >&2
      exit 0
    fi
    for f in $RUNTIME_CONFIGS; do
      [ -f "$keep/$f" ] || continue
      # 只在內容真的不同時才寫回並提示，避免每次部署都印一堆雜訊
      if ! cmp -s "$keep/$f" "$f" 2>/dev/null; then
        mkdir -p "$(dirname "$f")"
        cp -a "$keep/$f" "$f"
        echo "  保留現場設定: $f" >&2
      fi
    done
    rm -rf "$keep"
    ;;

  *)
    echo "用法: $0 save | $0 restore <暫存目錄>" >&2
    exit 2
    ;;
esac
