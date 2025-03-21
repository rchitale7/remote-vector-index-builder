# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import threading

class ResourceManager:
    def __init__(self, total_gpu_memory: float, total_cpu_memory: float):
        self._total_gpu_memory = total_gpu_memory
        self._total_cpu_memory = total_cpu_memory
        self._available_gpu_memory = total_gpu_memory
        self._available_cpu_memory = total_cpu_memory
        self._lock = threading.Lock()

    # TODO: separately track rejections due to GPU vs CPU
    def can_allocate(self, gpu_memory: float, cpu_memory: float) -> bool:
        with self._lock:
            return (self._available_gpu_memory >= gpu_memory and
                    self._available_cpu_memory >= cpu_memory)

    def allocate(self, gpu_memory: float, cpu_memory: float) -> bool:
        if not self.can_allocate(gpu_memory, cpu_memory):
            return False
        with self._lock:
            self._available_gpu_memory -= gpu_memory
            self._available_cpu_memory -= cpu_memory
            return True
        return False

    def release(self, gpu_memory: float, cpu_memory: float) -> None:
        with self._lock:
            self._available_gpu_memory += gpu_memory
            self._available_cpu_memory += cpu_memory

    def get_available_gpu_memory(self) -> float:
        with self._lock:
            return self._available_gpu_memory

    def get_available_cpu_memory(self) -> float:
        with self._lock:
            return self._available_cpu_memory