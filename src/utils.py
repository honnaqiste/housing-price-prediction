#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# -------------------- 日志配置 --------------------
# 不再在模块顶层调用 setup_logging，改为在 EnergyTracker 中按需配置

def setup_logging(log_to_file=True, log_level=logging.INFO, log_file_path=None):
    """
    配置日志：控制台始终输出，文件按需输出。
    如果 log_to_file 为 True 且 log_file_path 未指定，则自动生成带时间戳的默认路径。
    避免重复添加相同的 FileHandler。
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    root_logger = logging.getLogger()
    
    # 设置级别（如果尚未设置）
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(log_level)
    
    # 添加控制台 handler（如果还没有）
    has_console = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in root_logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
    
    # 文件日志处理
    if log_to_file:
        # 确定日志文件路径
        if log_file_path is None:
            # 默认路径：脚本所在父目录的父目录下的 logs 文件夹，文件名带时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_log_dir = Path(__file__).parent.parent / "logs"
            default_log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = default_log_dir / f"energy_measure_{timestamp}.log"
        else:
            log_file_path = Path(log_file_path)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经添加了指向同一文件的 FileHandler
        already_exists = False
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path.resolve()):
                already_exists = True
                break
        if not already_exists:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
            # 可选：输出一条日志表示文件路径
            logging.info(f"Logging to file: {log_file_path}")

# -------------------- EnergyTracker --------------------
class EnergyTracker:
    """
    RAPL-based energy tracker with background subtraction and real-time query.
    """

    def __init__(self, enable_background_removal=True, idle_duration=10.0,
                 verbose=True, domain=0, log_to_file=True, log_file_path=None):
        """
        :param log_to_file: 是否输出日志到文件
        :param log_file_path: 指定日志文件路径，若为 None 且 log_to_file=True，则自动生成带时间戳的默认路径
        """
        self.enable_background_removal = enable_background_removal
        self.idle_duration = idle_duration
        self.verbose = verbose
        self.domain = domain
        self.energy_path = f"/sys/class/powercap/intel-rapl:{domain}/energy_uj"
        self._check_available()
        self.background_power_watts = None
        self.start_energy_uj = None
        self.start_time = None
        self.background_subtracted = False
        self.net_energy_joules = 0.0
        
        # 配置日志：如果 log_to_file 为 True，则调用 setup_logging
        if log_to_file:
            setup_logging(log_to_file=True, log_file_path=log_file_path)
        elif not log_to_file and not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
            # 如果不需要文件日志，但控制台尚未配置，则只配置控制台
            setup_logging(log_to_file=False)

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