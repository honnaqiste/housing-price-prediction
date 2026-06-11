#!/bin/bash
# restore_system.sh - Restore services and settings that were stopped/disabled

echo "[RESTORE] Restoring system services and settings..."
sudo systemctl start gdm3
sudo systemctl start bluetooth
sudo systemctl start ssh
sudo systemctl start ssh.socket

# 1. Start stopped system services
echo "  Starting system services..."
sudo systemctl start bluetooth cups cups-browsed snapd unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
sudo systemctl start power-profiles-daemon 2>/dev/null || true
sudo systemctl start cron anacron 2>/dev/null || true
sudo systemctl start NetworkManager 2>/dev/null || true

# 2. Restore CPU governor to powersave or ondemand
echo "  Restoring CPU governor to powersave..."
sudo cpupower frequency-set -g powersave 2>/dev/null || echo "  cpupower not installed, skipping"

# 3. Restart user‑level indexers (optional)
echo "  Restarting file indexing services..."
systemctl --user start tracker-miner-fs 2>/dev/null || true
balooctl resume 2>/dev/null || true

# 4. Start some service
sudo systemctl start nxserver
sudo systemctl start docker
sudo systemctl stop docker.socket
# sudo systemctl start containerd 
# sudo systemctl start mihomo

echo "[RESTORE] System has been restored."
