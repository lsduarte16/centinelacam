#!/usr/bin/env bash
# =============================================================================
# RASPBERRY PI 5 INITIAL SETUP
# Run once on fresh Raspberry Pi OS (64-bit)
# =============================================================================
set -euo pipefail

echo "=== CAM-PI Gate Controller - Raspberry Pi 5 Setup ==="

# System updates
sudo apt update && sudo apt upgrade -y

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    sudo apt install -y docker-compose-plugin
fi

# Install Ollama for local LLM
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable ollama
    sudo systemctl start ollama
fi

# Pull lightweight model
echo "Pulling TinyLlama model..."
ollama pull tinyllama

# Create data directories
sudo mkdir -p /data/cam-pi/{logs,snapshots,models}
sudo chown -R "$USER:$USER" /data/cam-pi

# Clone repository
REPO_DIR="/opt/cam-pi"
if [ ! -d "$REPO_DIR" ]; then
    sudo mkdir -p "$REPO_DIR"
    sudo chown "$USER:$USER" "$REPO_DIR"
    git clone --depth 1 https://github.com/${GITHUB_REPO:-usuario/cam-pi}.git "$REPO_DIR"
fi

# Setup auto-update systemd timer
sudo tee /etc/systemd/system/cam-pi-update.service > /dev/null <<'EOF'
[Unit]
Description=CAM-PI Auto Update
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
User=pi
WorkingDirectory=/opt/cam-pi
ExecStart=/opt/cam-pi/scripts/auto_update.sh
Environment=GITHUB_REPO=usuario/cam-pi
EOF

sudo tee /etc/systemd/system/cam-pi-update.timer > /dev/null <<'EOF'
[Unit]
Description=CAM-PI Auto Update Timer

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
RandomizedDelaySec=30

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cam-pi-update.timer
sudo systemctl start cam-pi-update.timer

# Setup main service
sudo tee /etc/systemd/system/cam-pi.service > /dev/null <<'EOF'
[Unit]
Description=CAM-PI Gate Controller
After=network-online.target docker.service ollama.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/cam-pi
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cam-pi.service

# Enable hardware interfaces
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# GPU memory split for camera processing
if ! grep -q "gpu_mem=256" /boot/firmware/config.txt; then
    echo "gpu_mem=256" | sudo tee -a /boot/firmware/config.txt
fi

echo ""
echo "=== Setup Complete ==="
echo "1. Edit /opt/cam-pi/config/settings.yaml with your camera RTSP URL"
echo "2. Set GITHUB_REPO in /etc/systemd/system/cam-pi-update.service"
echo "3. Reboot: sudo reboot"
echo "4. Service starts automatically. Check: sudo systemctl status cam-pi"
echo ""
