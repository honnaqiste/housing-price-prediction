#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
from pathlib import Path

# -------------------- 日志配置 --------------------
LOG_FILE = Path(__file__).parent.parent / "logs" / "energy_measure.log"

def setup_logging(log_to_file=True, log_level=logging.INFO):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    if log_to_file:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE, encoding='utf-8'))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

setup_logging()

# -------------------- EnergyTracker --------------------
class EnergyTracker:
    """
    RAPL-based energy tracker with background subtraction and real-time query.
    """

    def __init__(self, enable_background_removal=True, idle_duration=10.0,
                 verbose=True, domain=0, log_to_file=True):
        self.enable_background_removal = enable_background_removal
        self.idle_duration = idle_duration
        self.verbose = verbose
        self.domain = domain
        self.energy_path = f"/sys/class/powercap/intel-rapl:{domain}/energy_uj"
        self._check_available()
        self.background_power_watts = None
        self.start_energy_uj = None
        self.start_time = None
        self.background_subtracted = False  # 标记是否已扣除背景
        self.net_energy_joules = 0.0
        if log_to_file:
            setup_logging(log_to_file=True)

    def _check_available(self):
        if not os.path.exists(self.energy_path):
            raise RuntimeError(f"RAPL interface not found: {self.energy_path}")

    def _read_energy_uj(self):
        with open(self.energy_path, 'r') as f:
            return int(f.read().strip())

    def measure_background_power(self, duration=None):
        if duration is None:
            duration = self.idle_duration
        e1 = self._read_energy_uj()
        time.sleep(duration)
        e2 = self._read_energy_uj()
        energy_joules = (e2 - e1) / 1_000_000.0
        avg_power = energy_joules / duration
        if self.verbose:
            logging.info(f"Background power measurement ({duration:.2f}s): {energy_joules:.3f} J, {avg_power:.3f} W")
        return avg_power

    def __enter__(self):
        if self.enable_background_removal:
            self.background_power_watts = self.measure_background_power()
        self.start_energy_uj = self._read_energy_uj()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 最终净能耗计算（可选，便于结束时获取）
        final_net = self.get_current_energy()
        self.net_energy_joules = final_net
        if self.verbose:
            logging.info(f"Final net energy: {final_net:.3f} J")

    def get_current_energy(self):
        """
        Returns net energy (J) from start to now, with background subtracted.
        Can be called multiple times during the measurement.
        """
        if self.start_energy_uj is None:
            raise RuntimeError("Called get_current_energy() before entering context")
        current_uj = self._read_energy_uj()
        elapsed_time = time.time() - self.start_time
        total_energy_joules = (current_uj - self.start_energy_uj) / 1_000_000.0

        if self.enable_background_removal and self.background_power_watts is not None:
            background_energy = self.background_power_watts * elapsed_time
            net = total_energy_joules - background_energy
            return max(net, 0.0)  # 避免负数
        else:
            return total_energy_joules

    def get_energy(self):
        """Return final net energy (only valid after exit)"""
        return self.net_energy_joules