#!/usr/bin/env bash
# 維運遠端 SSH 通道 —— ngrok TCP 轉本機 22。
#
# 🛑 免費方案的 TCP 位址「每次重啟都會變」,所以取得後一定要推 ntfy 通知,
#    否則機器重開機之後沒人知道新網址,等於失聯(現場曾因此要親自跑一趟)。
# 網址從 ngrok 自己的 logfmt 日誌取,不走 127.0.0.1:4040 API ——
# 這台同時有別的 ngrok agent(frm-ngrok),API 埠會被搶而浮動;
# 這版 ngrok(3.39) 也不支援 --web-addr 指定埠。
set -u

LOG="/tmp/ngrok-ssh.log"
NTFY_TOPIC="${NTFY_TOPIC:-field-jetson-default}"

: > "$LOG"
/usr/local/bin/ngrok tcp 22 --log=stdout --log-format=logfmt >> "$LOG" 2>&1 &
NGROK_PID=$!
trap 'kill "$NGROK_PID" 2>/dev/null' TERM INT

# 等 ngrok 註冊完並取得公開網址(最多 60 秒)
url=""
for _ in $(seq 30); do
  sleep 2
  url=$(grep -o 'url=tcp://[^ ]*' "$LOG" | tail -1 | cut -d= -f2-)
  [ -n "$url" ] && break
done

if [ -n "$url" ]; then
  hostport="${url#tcp://}"
  title="現場機 SSH 通道"
  msg="ssh -p ${hostport##*:} ubuntu@${hostport%%:*}"
else
  title="現場機 SSH 通道異常"
  msg="ngrok 已啟動但 60 秒內取不到公開網址,看 $LOG"
fi

curl -s --max-time 10 -H "Title: $title" -H "Priority: high" \
     -d "$msg" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
echo "$title: $msg"

wait "$NGROK_PID"
