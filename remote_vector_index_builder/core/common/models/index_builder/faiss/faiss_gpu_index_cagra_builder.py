# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from typing import Dict, Any
from dataclasses import field, dataclass

from core.common.models import (
    VectorsDataset,
    SpaceType,
)
from core.common.models.index_builder import (
    CagraGraphBuildAlgo,
    FaissGpuBuildIndexOutput,
    FaissGPUIndexBuilder,
)

from core.index_builder.index_builder_utils import configure_metric
from .ivf_pq_build_cagra_config import IVFPQBuildCagraConfig
from .ivf_pq_search_cagra_config import IVFPQSearchCagraConfig


@dataclass
class FaissGPUIndexCagraBuilder(FaissGPUIndexBuilder):
    """
    Configuration class for Faiss GPU Index Cagra
    Also exposes a method to build the gpu index from the CAGRA configuration
    """

    # Degree of input graph for pruning
    intermediate_graph_degree: int = 64
    # Degree of output graph
    graph_degree: int = 32
    # ANN Algorithm to build the knn graph
    graph_build_algo: CagraGraphBuildAlgo = CagraGraphBuildAlgo.IVF_PQ

    store_dataset: bool = False

    refine_rate: float = 2.0

    ivf_pq_build_config: IVFPQBuildCagraConfig = field(
        default_factory=IVFPQBuildCagraConfig
    )

    ivf_pq_search_config: IVFPQSearchCagraConfig = field(
        default_factory=IVFPQSearchCagraConfig
    )

    def _configure_build_algo(self):
        """
        Maps the graph building algorithm enum to the corresponding FAISS implementation.

        Args:
            graph_build_algo: The algorithm type to use for building the graph

        Returns:
            The corresponding FAISS graph building algorithm implementation
            Defaults to IVF_PQ if the specified algorithm is not found
        """
        switcher = {CagraGraphBuildAlgo.IVF_PQ: faiss.graph_build_algo_IVF_PQ}
        return switcher.get(self.graph_build_algo, faiss.graph_build_algo_IVF_PQ)

    def _validate_params(params: Dict[str, Any]) -> None:
        """
        Pre-validates FaissGPUIndexCagraBuilder configuration parameters before object creation.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If any parameter fails validation
        """
        if "intermediate_graph_degree" in params:
            if params["intermediate_graph_degree"] <= 0:
                raise ValueError(
                    "FaissGPUIndexCagraBuilder param: intermediate_graph_degree must be positive"
                )

        if "graph_degree" in params:
            if params["graph_degree"] <= 0:
                raise ValueError(
                    "FaissGPUIndexCagraBuilder param: graph_degree must be positive"
                )

        if "device" in params:
            if params["device"] < 0:
                raise ValueError(
                    "FaissGPUIndexCagraBuilder param: device must be non-negative"
                )

    def to_faiss_config(self) -> faiss.GpuIndexCagraConfig:
        """
        Builds and returns the complete faiss.GPUIndexCagraConfig
        Configures -
        - Basic GPUIndex Cagra Config parameters
        - IVF-PQ Build Cagra Config parameters
        - IVF-PQ Search Cagra Config paramters

        Returns:
            A fully configured faiss.GPUIndexCagraConfig object ready for index creation
        """

        gpu_index_cagra_config = faiss.GpuIndexCagraConfig()

        # Set basic parameters
        gpu_index_cagra_config.intermediate_graph_degree = (
            self.intermediate_graph_degree
        )
        gpu_index_cagra_config.graph_degree = self.graph_degree
        gpu_index_cagra_config.store_dataset = self.store_dataset
        gpu_index_cagra_config.device = self.device
        gpu_index_cagra_config.refine_rate = self.refine_rate

        # Set build algorithm
        gpu_index_cagra_config.build_algo = self._configure_build_algo()

        if self.graph_build_algo == CagraGraphBuildAlgo.IVF_PQ:
            gpu_index_cagra_config.ivf_pq_params = (
                self.ivf_pq_build_config.to_faiss_config()
            )
            gpu_index_cagra_config.ivf_pq_search_params = (
                self.ivf_pq_search_config.to_faiss_config()
            )

        return gpu_index_cagra_config

    @classmethod
    def from_dict(
        cls, params: Dict[str, Any] | None = None
    ) -> "FaissGPUIndexCagraBuilder":
        """
        Constructs a FaissGPUIndexCagraBuilder object from a dictionary of parameters.

        Args:
            params: A dictionary containing the configuration parameters

        Returns:
            A FaissGPUIndexCagraBuilder object with the specified configuration
        """
        if not params:
            return cls()

        # Create a copy of params to avoid modifying the original
        params_copy = params.copy()
        # Extract and configure IVF-PQ build parameters
        ivf_pq_params = params_copy.pop("ivf_pq_params", {})
        ivf_pq_build_config = IVFPQBuildCagraConfig.from_dict(ivf_pq_params)

        # Extract and configure IVF-PQ search parameters
        ivf_pq_search_params = params_copy.pop("ivf_pq_search_params", {})
        ivf_pq_search_config = IVFPQSearchCagraConfig.from_dict(ivf_pq_search_params)

        # Extract and configure graph build algo enum
        if "graph_build_algo" in params_copy:
            params_copy["graph_build_algo"] = CagraGraphBuildAlgo(
                params_copy["graph_build_algo"]
            )

        # Validate parameters
        cls._validate_params(params_copy)

        # Create and set the complete GPUIndexCagraConfig
        return cls(
            **params_copy,
            ivf_pq_build_config=ivf_pq_build_config,
            ivf_pq_search_config=ivf_pq_search_config,
        )

    def build_gpu_index(
        self,
        vectorsDataset: VectorsDataset,
        dataset_dimension: int,
        space_type: SpaceType,
    ) -> FaissGpuBuildIndexOutput:
        """
        Method to create a GPU Cagra Index to build a GPU Index for the specified vectors dataset

        Args:
        vectorsDataset (VectorsDataset): VectorsDataset object containing vectors and document IDs
        dataset_dimension (int): Dimension of the vectors
        space_type (SpaceType, optional): Distance metric to be used (defaults to L2)

        Returns:
        FaissGpuBuildIndexOutput: A data model containing the created GPU Index and dataset Vectors, Ids
        """
        faiss_gpu_index = None
        faiss_index_id_map = None
        faiss_gpu_index_config = None

        # Create a faiis equivalent version of gpu index build config
        try:
            faiss_gpu_index_config = self.to_faiss_config()
        except Exception as e:
            raise Exception(f"Failed to create faiss GPU index config: {str(e)}") from e

        try:
            # Configure the distance metric
            metric = configure_metric(space_type)

            res = faiss.StandardGpuResources()
            res.noTempMemory()
            # Create GPU CAGRA index with specified configuration
            faiss_gpu_index = faiss.GpuIndexCagra(
                res,
                dataset_dimension,
                metric,
                faiss_gpu_index_config,
            )

            # Create ID mapping layer to preserve document IDs
            faiss_index_id_map = faiss.IndexIDMap(faiss_gpu_index)
            # Add vectors and their corresponding IDs to the index
            faiss_index_id_map.add_with_ids(
                vectorsDataset.vectors, vectorsDataset.doc_ids
            )

            return FaissGpuBuildIndexOutput(
                gpu_index=faiss_gpu_index, index_id_map=faiss_index_id_map
            )
        except Exception as e:
            if faiss_gpu_index is not None:
                faiss_gpu_index.thisown = True
                faiss_gpu_index.__swig_destroy__(faiss_gpu_index)
            if faiss_index_id_map is not None:
                faiss_index_id_map.thisown = True
                faiss_index_id_map.own_fields = False
                faiss_index_id_map.__swig_destroy__(faiss_index_id_map)
            raise Exception(f"Failed to create faiss GPU index: {str(e)}") from e
