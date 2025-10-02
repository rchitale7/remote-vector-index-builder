# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from py3nvml import nvidia_smi
import psutil
from datetime import datetime
import threading
import pandas as pd
import time
import logging
import os

logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, filename, interval: float = 0.1, monitor_gpu=True):
        self.interval = interval
        self.cpu_memory_logs = []
        self.gpu_memory_logs = []
        self.gpu_process_level_logs = []
        self.is_monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
        self.filename = filename.split("/")[-1]

        # Initialize GPU monitoring
        if monitor_gpu:
            self.monitor_gpu = True
            self.gpu_id = 0
            self.gpu_memory_logs = []
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu_id)
        else:
            self.monitor_gpu = False

    def _get_cpu_system_memory_info(self):
        """Get system CPU memory usage in MB"""
        cpu_info = psutil.virtual_memory()
        return cpu_info.used / 1024 / 1024

    def _get_gpu_system_memory_info(self):
        """Get system GPU memory usage in MB"""
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / 1024 / 1024

    def _get_cpu_process_memory_info(self) -> float:
        """Get current process memory usage in MB"""
        return self.process.memory_info().rss / (1024**2)  # Convert bytes to MB

    def _get_gpu_process_memory_info(self):
        """"Aggregate GPU processes memory usage in MB"""
        compute_procs = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(self.gpu_id)
        process_logs = []
        total_memory = 0
        if compute_procs:
            for p in compute_procs:
                used_memory = p.usedGpuMemory / 1024 / 1024
                process_logs.append(
                    {
                        "pid": p.pid,
                        "name": self._get_process_name(p.pid),
                        "used_memory": used_memory
                    }
                )
                total_memory += used_memory

        return process_logs, total_memory

    def _get_process_name(self, pid):
        """Get the name of a process given its PID."""
        try:
            process = psutil.Process(pid)
            return process.name()
        except psutil.NoSuchProcess:
            return "Unknown"


    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:

            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.monitor_gpu:
                system_gpu_mb = self._get_gpu_system_memory_info()

                self.gpu_memory_logs.append(
                    {
                        "gpu_used_system_memory": system_gpu_mb,
                        "cur_time": cur_time
                    }
                )

            process_cpu_mb = self._get_cpu_process_memory_info()
            system_cpu_mb = self._get_cpu_system_memory_info()
            self.cpu_memory_logs.append(
                {
                    "cpu_used_system_memory": system_cpu_mb,
                    "cpu_used_process_memory": process_cpu_mb,
                    "cur_time": cur_time
                }
            )
            time.sleep(self.interval)  # sampling rate

    def start_monitoring(self):
        """Start GPU memory monitoring"""
        self.is_monitoring = True
        self.gpu_memory_logs = []
        self.cpu_memory_logs = []
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop GPU memory monitoring"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def log_system_gpu_metrics(self):
        """Log GPU memory usage metrics"""
        if self.monitor_gpu:
            df = pd.DataFrame(self.gpu_memory_logs)
            max_memory = df["gpu_used_system_memory"].max()
            start_memory = df["gpu_used_system_memory"].iloc[0]
            end_memory = df["gpu_used_system_memory"].iloc[-1]
            logger.debug(f"Start system GPU Memory: ,{start_memory}")
            logger.debug(f"End system GPU Memory: ,{end_memory}")
            logger.debug(f"Max system GPU Memory: ,{max_memory}")
            logger.debug(f"Net system GPU Memory used:, {max_memory-start_memory}")

            # max_process_memory = df["gpu_used_agg_process_memory"].max()
            # start_process_memory = df["gpu_used_agg_process_memory"].iloc[0]
            # end_process_memory = df["gpu_used_agg_process_memory"].iloc[-1]
            # logger.debug(f"Start process GPU Memory: ,{start_process_memory}")
            # logger.debug(f"End process GPU Memory: ,{end_process_memory}")
            # logger.debug(f"Max process GPU Memory: ,{max_process_memory}")
            # logger.debug(f"Net process GPU Memory used:, {max_process_memory-start_process_memory}")

            df.to_csv(f'/files/gpu_metrics_{self.filename}.csv')
            return max_memory, start_memory, end_memory
        return 0, 0, 0

    def log_system_cpu_metrics(self):
        """Log CPU memory usage metrics"""
        df = pd.DataFrame(self.cpu_memory_logs)
        max_memory = df["cpu_used_system_memory"].max()
        start_memory = df["cpu_used_system_memory"].iloc[0]
        end_memory = df["cpu_used_system_memory"].iloc[-1]
        logger.debug(f"Start CPU Memory: ,{start_memory}")
        logger.debug(f"End CPU Memory: ,{end_memory}")
        logger.debug(f"Max CPU Memory: ,{max_memory}")
        logger.debug(f"Net CPU Memory used:, {max_memory-start_memory}")

        max_process_memory = df["cpu_used_process_memory"].max()
        start_process_memory = df["cpu_used_process_memory"].iloc[0]
        end_process_memory = df["cpu_used_process_memory"].iloc[-1]
        logger.debug(f"Start process CPU Memory: ,{start_process_memory}")
        logger.debug(f"End process CPU Memory: ,{end_process_memory}")
        logger.debug(f"Max process CPU Memory: ,{max_process_memory}")
        logger.debug(f"Net process CPU Memory used:, {max_process_memory-start_process_memory}")

        df.to_csv(f'/files/cpu_metrics_{self.filename}.csv')

        return max_memory, start_memory, end_memory

    def __del__(self):
        """Cleanup NVML on object destruction"""
        try:
            if self.monitor_gpu:
                nvidia_smi.nvmlShutdown()
        except Exception:
            pass