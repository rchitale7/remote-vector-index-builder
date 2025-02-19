# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel
from schemas.api import CreateJobRequest

class BuildWorkflow(BaseModel):
    job_id: str
    gpu_memory_required: float
    cpu_memory_required: float
    create_job_request: CreateJobRequest
