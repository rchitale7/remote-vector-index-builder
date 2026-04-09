# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from typing import Dict, Any

from core.common.models.index_builder.faiss import FaissIndexHNSWCagraBuilder
from core.common.models.index_builder import (
    FaissGpuBuildIndexOutput,
    FaissCpuBuildIndexOutput,
)


class TestFaissIndexHNSWCagraBuilder:

    @pytest.fixture
    def default_builder(self, request):
        return FaissIndexHNSWCagraBuilder()

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {"ef_search": 200, "ef_construction": 150, "base_level_only": False}

    @pytest.fixture
    def mock_gpu_index(self):
        """Mock GPU index with tracking"""
        gpu_index = faiss.GpuIndexCagra()
        return gpu_index

    @pytest.fixture
    def mock_index_id_map(self):
        """Mock index ID map"""
        return faiss.IndexIDMap()

    def test_default_initialization(self, default_builder):
        assert default_builder.ef_search == 100
        assert default_builder.ef_construction == 100
        assert default_builder.base_level_only is True

    def test_custom_initialization(self, custom_params):
        custom_builder = FaissIndexHNSWCagraBuilder(**custom_params)
        assert custom_builder.ef_search == custom_params["ef_search"]
        assert custom_builder.ef_construction == custom_params["ef_construction"]
        assert custom_builder.base_level_only == custom_params["base_level_only"]

    def test_from_dict_partial(self):
        partial_params = {"ef_search": 200, "ef_construction": 158}
        builder = FaissIndexHNSWCagraBuilder.from_dict(partial_params)
        assert builder.ef_search == 200
        assert builder.ef_construction == 158
        assert builder.base_level_only is True  # default value

    def test_convert_gpu_to_cpu_index_success(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        """Test successful GPU to CPU index conversion"""
        gpu_build_output = FaissGpuBuildIndexOutput(
            gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
        )

        # Perform conversion
        result = default_builder.convert_gpu_to_cpu_index(gpu_build_output)

        # Verify result type and structure
        assert isinstance(result, FaissCpuBuildIndexOutput)
        assert isinstance(result.cpu_index, faiss.IndexHNSWCagra)
        assert isinstance(result.index_id_map, faiss.IndexIDMap)

        # Verify CPU index configuration
        assert result.cpu_index.hnsw.efConstruction == default_builder.ef_construction
        assert result.cpu_index.hnsw.efSearch == default_builder.ef_search
        assert result.cpu_index.base_level_only == default_builder.base_level_only

        assert gpu_build_output.index_id_map is None

    def test_convert_gpu_to_cpu_index_copy_error(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        def failing_copy(*args):
            raise RuntimeError("Simulated copy error")

        mock_gpu_index.copyTo = failing_copy

        with pytest.raises(Exception) as exc_info:
            default_builder.convert_gpu_to_cpu_index(
                FaissGpuBuildIndexOutput(
                    gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
                )
            )
        assert "Failed to convert GPU index to CPU index" in str(exc_info.value)
        assert "Simulated copy error" in str(exc_info.value)

    def test_skip_stored_vectors_passes_skip_storage(
        self, mock_gpu_index, mock_index_id_map
    ):
        """Test that skip_stored_vectors=True passes skip_storage=True to copyTo"""
        builder = FaissIndexHNSWCagraBuilder(skip_stored_vectors=True)

        copy_calls = []
        original_copyTo = mock_gpu_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is True

    def test_skip_stored_vectors_false_passes_skip_storage_false(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        """Test that skip_stored_vectors=False (default) passes skip_storage=False to copyTo"""
        copy_calls = []
        original_copyTo = mock_gpu_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_index.copyTo = tracking_copyTo

        default_builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is False

    def test_binary_skip_stored_vectors_passes_skip_storage(self):
        """Test that skip_stored_vectors=True passes skip_storage=True to binary copyTo"""
        from core.common.models.index_build_parameters import DataType

        builder = FaissIndexHNSWCagraBuilder(
            skip_stored_vectors=True, vector_dtype=DataType.BINARY
        )

        mock_gpu_binary_index = faiss.GpuIndexBinaryCagra()
        mock_binary_id_map = faiss.IndexBinaryIDMap()

        copy_calls = []
        original_copyTo = mock_gpu_binary_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_binary_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_binary_index, index_id_map=mock_binary_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is True

    def test_binary_skip_stored_vectors_false_does_not_skip_storage(self):
        """Test that skip_stored_vectors=False (default) does not skip storage for binary copyTo"""
        from core.common.models.index_build_parameters import DataType

        builder = FaissIndexHNSWCagraBuilder(vector_dtype=DataType.BINARY)

        mock_gpu_binary_index = faiss.GpuIndexBinaryCagra()
        mock_binary_id_map = faiss.IndexBinaryIDMap()

        copy_calls = []
        original_copyTo = mock_gpu_binary_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_binary_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_binary_index, index_id_map=mock_binary_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is False
