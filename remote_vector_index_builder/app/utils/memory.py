# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from core.common.models import IndexBuildParameters


def calculate_memory_requirements(
    vector_dimensions: int,
    num_vectors: int,
    index_build_parameters: IndexBuildParameters,
) -> tuple[float, float]:
    """
    Calculate GPU and CPU memory requirements for a vector workload.

    This function estimates the memory needed for processing vector operations,
    taking into account the workload size and complexity.

    Note:
    The memory estimations are specific to building a CAGRA index on GPUs,
    and converting the CAGRA index to CPU compatible HNSW. The GPU memory calculation is a very rough estimate.
    There is an open issue with NVIDIA on how to better calculate memory taken up
    by a CAGRA index: https://github.com/rapidsai/cuvs/issues/566

    Args:
        vector_dimensions: Vector dimensions
        num_vectors: Number of vectors
        index_build_parameters: Build parameters that contain value of m and vector data type

    Returns:
        tuple[float, float]: A tuple containing:
            - gpu_memory (float): Required GPU memory in bytes
            - cpu_memory (float): Required CPU memory in bytes
    """

    m = index_build_parameters.algorithm_parameters.m
    entry_size = index_build_parameters.data_type.get_size()

    # Vector memory (same for both GPU and CPU)
    vector_memory = vector_dimensions * num_vectors * entry_size

    # use formula to calculate memory taken up by index
    index_gpu_memory = (
        (vector_dimensions * entry_size + m * 8) * 1.1 * num_vectors
    ) * 1.5

    index_cpu_memory = (vector_dimensions * entry_size + m * 8) * 1.1 * num_vectors

    return (index_gpu_memory + vector_memory), (index_cpu_memory + vector_memory)
