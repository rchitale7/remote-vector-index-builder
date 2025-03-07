# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel
from core.common.models.index_build_parameters import IndexBuildParameters

class BuildWorkflow(BaseModel):
    job_id: str
    gpu_memory_required: float
    cpu_memory_required: float
    index_build_parameters: IndexBuildParameters
