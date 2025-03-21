# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from enum import Enum
from pydantic import BaseModel
from app.models.request import RequestParameters
from typing import Optional

class JobStatus(str, Enum):
    RUNNING = "RUNNING_INDEX_BUILD"
    FAILED = "FAILED_INDEX_BUILD"
    COMPLETED = "COMPLETED_INDEX_BUILD"

class Job(BaseModel):
    id: str
    status: JobStatus
    request_parameters: RequestParameters
    file_path: Optional[str] = None
    msg: Optional[str] = None

    def compare_request_parameters(self, other: RequestParameters) -> bool:
        return self.request_parameters == other