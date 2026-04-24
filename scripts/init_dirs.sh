#!/usr/bin/env bash
# 第一次 clone 後跑一次，建好 data/ models/ output/ storage/
#
# 用法：
#   1) 沒有外接硬碟（直接用本機）：
#        bash scripts/init_dirs.sh
#
#   2) 有 NVMe / 外接硬碟，要把資料放在別的 mount：
#        TRAFFIC_STORAGE_ROOT=/mnt/nvme/traffic bash scripts/init_dirs.sh
#      會自動建立 symlink：data/models/output/storage → $TRAFFIC_STORAGE_ROOT/<name>
#
# 重跑安全：已經存在的目錄或 symlink 會跳過，不會覆寫資料。

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT="$(pwd)"

DIRS=(data models output storage)
ROOT="${TRAFFIC_STORAGE_ROOT:-}"

if [[ -n "$ROOT" ]]; then
  echo "📦 外接硬碟模式：$ROOT"
  mkdir -p "$ROOT"
  for d in "${DIRS[@]}"; do
    mkdir -p "$ROOT/$d"
    if [[ -e "$PROJECT/$d" && ! -L "$PROJECT/$d" ]]; then
      echo "⚠️  $d 已經是實體目錄，跳過（若要搬到 $ROOT 請手動移動後再跑）"
      continue
    fi
    ln -sfn "$ROOT/$d" "$PROJECT/$d"
    echo "🔗 $d → $ROOT/$d"
  done
else
  echo "💾 本機模式（沒設 TRAFFIC_STORAGE_ROOT）"
  for d in "${DIRS[@]}"; do
    if [[ -L "$PROJECT/$d" ]]; then
      target="$(readlink "$PROJECT/$d")"
      if [[ ! -e "$PROJECT/$d" ]]; then
        echo "❌ $d 是壞掉的 symlink（指向 $target），砍掉改建實體目錄"
        rm -f "$PROJECT/$d"
        mkdir -p "$PROJECT/$d"
        echo "✅ $d (new empty dir)"
      else
        echo "🔗 $d → $target (既有 symlink 維持)"
      fi
    elif [[ ! -e "$PROJECT/$d" ]]; then
      mkdir -p "$PROJECT/$d"
      echo "✅ $d (new empty dir)"
    else
      echo "📁 $d 已存在"
    fi
  done
fi

echo ""
echo "完成。檢查："
ls -la "${DIRS[@]/#/$PROJECT/}" 2>/dev/null | grep -E "^(l|d)" | awk '{print "  " $NF, ($1 ~ /^l/ ? "->" $NF : "")}' || true
