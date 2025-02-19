# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from schemas.api import DataType

def calculate_memory_requirements(
        vector_dimensions: int,
        num_vectors: int,
        data_type: DataType,
        m: int
) -> tuple[float, float]:
    """
        Calculate GPU and CPU memory requirements for a vector workload.

        This function estimates the memory needed for processing vector operations,
        taking into account the workload size and complexity.

        Returns:
        tuple[float, float]: A tuple containing:
            - gpu_memory_gb (float): Required GPU memory in gigabytes
            - cpu_memory_gb (float): Required CPU memory in gigabytes
    """

    if data_type == DataType.FLOAT32:
        entry_size = 4
    elif data_type == DataType.FLOAT16:
        entry_size = 2
    elif data_type == DataType.BYTE:
        entry_size = 1
    elif data_type == DataType.BINARY:
        entry_size=0.125
    # Vector memory (same for both GPU and CPU)
    vector_memory = (vector_dimensions * num_vectors * entry_size) / (2 ** 30)  # 4 bytes per float32

    # use formula to calculate memory taken up by index
    index_gpu_memory = (((vector_dimensions * entry_size + m*8) * 1.1 * num_vectors) / (2 ** 30)) * 1.5

    index_cpu_memory = ((vector_dimensions * entry_size + m*8) * 1.1 * num_vectors) / (2 ** 30)

    return (index_gpu_memory + vector_memory), (index_cpu_memory + vector_memory)