#!/bin/sh
# tvd-proxy(對外 HTTPS 反向代理,nginx:alpine)的啟動參數。
#
# 為什麼要有這支:這個容器一直是手動 `docker run` 起來的,參數只存在於
# 現場那台機器的 docker 裡 —— 沒有 compose、沒有版控。一旦誰把它砍掉重建,
# 所有非預設的參數就跟著消失,而且不會有人發現。
#
# 🛑 2026-08-18 就踩到了:容器沒有帶 TZ,nginx 的 $time_local 用 UTC,
#    對外存取日誌整整慢 8 小時。主機時間是對的(NTP 同步 10.41.0.111,
#    偏差 -1.8ms),錯的只有容器 —— 這種錯最難發現,因為 docker logs 前面
#    那個時間戳是 docker daemon 加的(主機時間,正確),只有 log 行「裡面」
#    那個 [18/Aug/2026:03:59:31 +0000] 是錯的。
#    nginx:alpine 內含 tzdata,所以 TZ=Asia/Taipei 直接可用,不必像
#    某些精簡映像要退而求其次寫 TZ=UTC-8。
#
# 用法(在現場機上):
#     PROXY_HOME=/home/ubuntu/tvd-proxy sh scripts/field/tvd-proxy-run.sh
# 預設 PROXY_HOME 取當前使用者家目錄下的 tvd-proxy。
#   87  = /home/ubuntu/tvd-proxy
#   104 = /home/mic-711/tvd-proxy
#
# 重建流程(host network 不能同時綁同一個埠,所以要先停舊的):
#     docker stop tvd-proxy && docker rename tvd-proxy tvd-proxy-old
#     sh scripts/field/tvd-proxy-run.sh
#     # 驗過再 docker rm tvd-proxy-old
set -e
PROXY_HOME="${PROXY_HOME:-$HOME/tvd-proxy}"

[ -f "$PROXY_HOME/nginx.conf" ] || { echo "找不到 $PROXY_HOME/nginx.conf"; exit 1; }
[ -d "$PROXY_HOME/certs" ] || { echo "找不到 $PROXY_HOME/certs"; exit 1; }

docker run -d \
    --name tvd-proxy \
    --restart unless-stopped \
    --network host \
    -e TZ=Asia/Taipei \
    -v "$PROXY_HOME/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -v "$PROXY_HOME/certs:/etc/nginx/certs:ro" \
    nginx:alpine

sleep 5
echo "容器時間: $(docker exec tvd-proxy date '+%F %T %Z')"
echo "主機時間: $(date '+%F %T %Z')"
curl -sk -m 8 -o /dev/null -w 'https 8443 -> HTTP %{http_code}\n' https://127.0.0.1:8443/api/health
