#!/bin/bash
# prepare_measurement.sh - Stop unnecessary services and processes before measuring energy consumption

set -e  # Exit on error

echo "[PREPARE] Starting system cleanup for energy measurement..."

# 1. Kill common browser processes
echo "  Closing browsers..."
pkill -f "firefox" 2>/dev/null || true
pkill -f "chrome" 2>/dev/null || true
pkill -f "chromium" 2>/dev/null || true
pkill -f "microsoft-edge" 2>/dev/null || true

# 2. Stop user‑level background indexers
echo "  Stopping file indexing services..."
systemctl --user stop tracker-miner-fs 2>/dev/null || true
systemctl --user stop tracker-extract 2>/dev/null || true
balooctl suspend 2>/dev/null || true

# 3. Stop system services (requires sudo)
echo "  Stopping system services (requires sudo)..."
sudo systemctl stop bluetooth cups cups-browsed snapd unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
sudo systemctl stop power-profiles-daemon 2>/dev/null || true

# Optional: also stop NetworkManager if networking is not needed
sudo systemctl stop NetworkManager
sudo systemctl stop bluetooth

# 4. Temporarily disable cron jobs
echo "  Disabling cron/anacron temporary jobs..."
sudo systemctl stop cron anacron 2>/dev/null || true

# 5. Set CPU performance mode to 'performance' (reduce frequency scaling noise)
echo "  Setting CPU governor to performance..."
sudo cpupower frequency-set -g performance 2>/dev/null || echo "  cpupower not installed, skipping"

# 6. Run powertop --auto-tune to optimise power management (lower idle power consumption)
if command -v powertop &> /dev/null; then
    echo "  Running powertop --auto-tune ..."
    sudo powertop --auto-tune
else
    echo "  powertop not installed, skipping auto‑tune"
fi

# 7. Advise user to switch to a TTY
echo "[PREPARE] Done! It is recommended to switch to a TTY (Ctrl+Alt+F3) before running the training script."
echo "         If you switch, re‑run this script afterwards – it will clean up again (browsers will be gone without GUI)."
sudo systemctl stop gdm3
