#!/bin/sh
# MiiNePort(10.42.38.35)單播繞道 —— 持續校正版。
# 見 /etc/systemd/system/miineport-route.service 的說明:
# 同一個 IP 在兩個網段各有一台設備,ARP 被攪動後會打到錯的那台,
# 所以不能只在開機套用一次。
IFACE=enP5p4s0
SRC=10.42.38.200
TARGET=10.42.38.35
MAC=00:90:e8:89:11:42
INTERVAL=30

ip addr add "$SRC/32" dev "$IFACE" 2>/dev/null
echo "[miineport-route] 啟動:$TARGET via $IFACE src $SRC mac $MAC"

while :; do
    ip route replace "$TARGET/32" dev "$IFACE" src "$SRC" 2>/dev/null
    cur=$(ip neigh show "$TARGET" dev "$IFACE" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="lladdr") print $(i+1)}')
    if [ "$cur" != "$MAC" ]; then
        echo "[miineport-route] ARP 不對(${cur:-無}) → 修回 $MAC"
        ip neigh replace "$TARGET" lladdr "$MAC" dev "$IFACE" nud permanent
    fi
    # 相機網段那台同 IP 的設備不要干擾我們:把它在相機口的 ARP 清掉,
    # 避免核心挑到那條。不影響相機本身(10.42.40.22~25 各自的 ARP 不動)。
    ip neigh del "$TARGET" dev enP5p5s0 2>/dev/null
    sleep "$INTERVAL"
done
