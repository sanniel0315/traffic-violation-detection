#!/bin/sh
# MiiNePort(10.42.38.35)單播繞道 —— ⚠️ 2026-08-18 起「預設不需要、也不該啟用」。
#
# 這支腳本是為了 2026-08-17 當下的現場接線寫的:號誌那顆 MiiNePort 掛在
# enP5p4s0(192.168.1.x)這一段,但它自己的 IP/遮罩/閘道是 10.42.38.35 /20 →
# 10.42.32.254,回覆封包會查自己的路由表送去那個不在本廣播域的閘道而回不來。
# 解法是在 enP5p4s0 補一個 10.42.38.200/32 的來源位址 + /32 主機路由 + 靜態 ARP。
#
# 🛑 2026-08-18 現場把號誌那條線改接進 10.42.32.0/20(跟相機同一段,走 enP5p5s0)。
#    此後這支腳本從「救援」變成「破壞」—— 它每 30 秒會:
#      ① 把 10.42.38.35/32 釘回 enP5p4s0(那一段已經沒有這台設備)
#      ② 灌 nud permanent 的靜態 ARP(設備不在,封包石沉大海也不會報錯)
#      ③ ip neigh del 10.42.38.35 dev enP5p5s0 —— 主動刪掉「正確那一側」的 ARP
#    症狀:ping 100% 丟包、抄錄器一直 TimeoutError,但 link/路由/ARP 表看起來都「正常」。
#    當天實際查法:Moxa UDP 4800 廣播探測分別綁兩個介面 ——
#      enP5p4s0 找到 0 台 / enP5p5s0 找到 69 台(含 00:90:e8:89:11:42 = 10.42.38.35)。
#    L2 廣播探測不受 IP/路由設定影響,是判斷「設備到底掛在哪一段」最直接的工具。
#
# 現在 87 上已 systemctl disable --now,保留這支只為了「線又被改回 192.168.1.x」時能救。
# 下面的自檢會在正常路由已經通得到設備時直接退出,不再動任何設定。
IFACE=enP5p4s0
SRC=10.42.38.200
TARGET=10.42.38.35
MAC=00:90:e8:89:11:42
INTERVAL=30

# ── 自檢:正常路由已經到得了就什麼都不要做 ─────────────────────────────
# 沒有這道閘,誤啟用這支服務會把已經好好的連線弄壞(2026-08-18 實際發生過)。
if ping -c 2 -W 2 "$TARGET" >/dev/null 2>&1; then
    dev=$(ip route get "$TARGET" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
    if [ "$dev" != "$IFACE" ]; then
        echo "[miineport-route] $TARGET 已能經 ${dev:-預設路由} 直接連通 —— 不需要繞道,結束"
        exit 0
    fi
fi

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
    # 🛑 這一行就是 2026-08-18 那次的元凶 —— 線改段之後「錯的那台」才是對的那台。
    ip neigh del "$TARGET" dev enP5p5s0 2>/dev/null
    sleep "$INTERVAL"
done
