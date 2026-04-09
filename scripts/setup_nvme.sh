#!/bin/bash
set -e

echo "=== NVMe SSD 掛載與資料遷移腳本 ==="

# 1. 掛載
echo "[1/6] 掛載 NVMe SSD..."
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p2 /mnt/nvme
sudo chown -R ubuntu:ubuntu /mnt/nvme
df -h /mnt/nvme
echo "OK: 掛載成功"

# 2. 寫入 fstab 開機自動掛載
echo "[2/6] 設定開機自動掛載..."
UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p2)
if ! grep -q "$UUID" /etc/fstab; then
    echo "UUID=$UUID /mnt/nvme ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab
    echo "OK: 已寫入 /etc/fstab"
else
    echo "OK: fstab 已存在此條目"
fi

# 3. 建立目錄結構
echo "[3/6] 建立 NVMe 目錄結構..."
mkdir -p /mnt/nvme/traffic/{models,storage,data,output,docker}
echo "OK: 目錄結構建立完成"

# 4. 遷移資料
PROJECT="/home/ubuntu/traffic-violation-detection"
echo "[4/6] 遷移資料到 NVMe..."

# models
if [ -d "$PROJECT/models" ] && [ ! -L "$PROJECT/models" ]; then
    cp -a "$PROJECT/models/"* /mnt/nvme/traffic/models/ 2>/dev/null || true
    mv "$PROJECT/models" "$PROJECT/models.bak"
    ln -sf /mnt/nvme/traffic/models "$PROJECT/models"
    echo "  models -> /mnt/nvme/traffic/models"
fi

# storage
if [ -d "$PROJECT/storage" ] && [ ! -L "$PROJECT/storage" ]; then
    cp -a "$PROJECT/storage/"* /mnt/nvme/traffic/storage/ 2>/dev/null || true
    mv "$PROJECT/storage" "$PROJECT/storage.bak"
    ln -sf /mnt/nvme/traffic/storage "$PROJECT/storage"
    echo "  storage -> /mnt/nvme/traffic/storage"
fi

# data (database)
if [ -d "$PROJECT/data" ] && [ ! -L "$PROJECT/data" ]; then
    cp -a "$PROJECT/data/"* /mnt/nvme/traffic/data/ 2>/dev/null || true
    mv "$PROJECT/data" "$PROJECT/data.bak"
    ln -sf /mnt/nvme/traffic/data "$PROJECT/data"
    echo "  data -> /mnt/nvme/traffic/data"
fi

# output
if [ -d "$PROJECT/output" ] && [ ! -L "$PROJECT/output" ]; then
    cp -a "$PROJECT/output/"* /mnt/nvme/traffic/output/ 2>/dev/null || true
    mv "$PROJECT/output" "$PROJECT/output.bak"
    ln -sf /mnt/nvme/traffic/output "$PROJECT/output"
    echo "  output -> /mnt/nvme/traffic/output"
fi

echo "OK: 資料遷移完成"

# 5. 遷移 Docker data-root（釋放 eMMC 最大空間）
echo "[5/6] 遷移 Docker data..."
if [ ! -f /etc/docker/daemon.json ] || ! grep -q "nvme" /etc/docker/daemon.json 2>/dev/null; then
    sudo systemctl stop docker 2>/dev/null || true
    sudo mkdir -p /mnt/nvme/traffic/docker
    if [ -d /var/lib/docker ] && [ ! -L /var/lib/docker ]; then
        sudo rsync -aP /var/lib/docker/ /mnt/nvme/traffic/docker/
    fi
    sudo mkdir -p /etc/docker
    echo '{"data-root":"/mnt/nvme/traffic/docker"}' | sudo tee /etc/docker/daemon.json
    sudo systemctl start docker
    echo "OK: Docker data-root -> /mnt/nvme/traffic/docker"
else
    echo "OK: Docker 已設定使用 NVMe"
fi

# 6. 驗證
echo "[6/6] 驗證結果..."
echo "=== 磁碟使用 ==="
df -h / /mnt/nvme
echo ""
echo "=== Symlinks ==="
ls -la "$PROJECT/models" "$PROJECT/storage" "$PROJECT/data" "$PROJECT/output" 2>/dev/null
echo ""
echo "=== NVMe 使用 ==="
du -sh /mnt/nvme/traffic/*
echo ""
echo "✅ 全部完成！NVMe SSD 已就緒。"
