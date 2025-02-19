# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class DataType(str, Enum):
    FLOAT32 = 'fp32'
    FLOAT16 = 'fp16'
    BYTE = 'byte'
    BINARY='binary'

class SpaceType(str, Enum):
    L2 = "l2"
    INNERPRODUCT = "innerproduct"

class AlgorithmParameters(BaseModel):
    ef_construction: int = 128
    m: int = 16

class IndexParameters(BaseModel):
    algorithm: str = "hnsw"
    space_type: SpaceType = SpaceType.L2
    algorithm_parameters: AlgorithmParameters = Field(
        default_factory=AlgorithmParameters
    )

class CreateJobRequest(BaseModel):
    repository_type: str
    container_name: str
    object_path: str
    dimension: int
    doc_count: int
    tenant_id: str = ""
    data_type: DataType = DataType.FLOAT32
    engine: str = "faiss"
    index_parameters: IndexParameters = Field(
        default_factory=IndexParameters
    )

    class Config:
        extra = "forbid"

class CreateJobResponse(BaseModel):
    job_id: str

class GetStatusResponse(BaseModel):
    task_status: str
    knn_index_path: Optional[str] = None
    msg: Optional[str] = None