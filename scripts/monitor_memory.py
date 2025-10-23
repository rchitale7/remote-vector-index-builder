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
import argparse


class MemoryMonitor:
    def __init__(self, identifier, interval: float = 0.1, monitor_gpu=True):
        self.interval = interval
        self.cpu_memory_logs = []
        self.gpu_memory_logs = []
        self.gpu_process_level_logs = []
        self.is_monitoring = False
        self._monitor_thread = None
        self.process = psutil.Process()
        self.identifier = identifier

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

            system_cpu_mb = self._get_cpu_system_memory_info()
            self.cpu_memory_logs.append(
                {
                    "cpu_used_system_memory": system_cpu_mb,
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
            logging.info(f"Start system GPU Memory: ,{start_memory}")
            logging.info(f"End system GPU Memory: ,{end_memory}")
            logging.info(f"Max system GPU Memory: ,{max_memory}")
            logging.info(f"Net system GPU Memory used:, {max_memory-start_memory}")

            df.to_csv(f'./gpu_stats_{self.identifier}.csv')
            return max_memory, start_memory, end_memory
        return 0, 0, 0

    def log_system_cpu_metrics(self):
        """Log CPU memory usage metrics"""
        df = pd.DataFrame(self.cpu_memory_logs)
        max_memory = df["cpu_used_system_memory"].max()
        start_memory = df["cpu_used_system_memory"].iloc[0]
        end_memory = df["cpu_used_system_memory"].iloc[-1]
        logging.info(f"Start CPU Memory: ,{start_memory}")
        logging.info(f"End CPU Memory: ,{end_memory}")
        logging.info(f"Max CPU Memory: ,{max_memory}")
        logging.info(f"Net CPU Memory used:, {max_memory-start_memory}")

        df.to_csv(f'./cpu_stats_{self.identifier}.csv')

        return max_memory, start_memory, end_memory

    def __del__(self):
        """Cleanup NVML on object destruction"""
        try:
            if self.monitor_gpu:
                nvidia_smi.nvmlShutdown()
        except Exception:
            pass

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor system memory usage')
    parser.add_argument('--identifier', required=True, help='Identifier for output files')
    parser.add_argument('--interval', type=float, default=0.1, help='Monitoring interval in seconds')
    parser.add_argument('--monitor-gpu', action='store_true', help='Enable GPU monitoring')

    args = parser.parse_args()
    setup_logging()

    monitor = MemoryMonitor(args.identifier, args.interval, args.monitor_gpu)
    try:
        monitor.start_monitoring()
        input("Press Enter to stop monitoring...")
    finally:
        monitor.stop_monitoring()
        monitor.log_system_cpu_metrics()
        monitor.log_system_gpu_metrics()
