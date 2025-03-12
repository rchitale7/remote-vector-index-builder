# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class CreateJobResponse(BaseModel):
    job_id: str

# TODO: move this core image
class GetStatusResponse(BaseModel):
    task_status: str
    file_name: Optional[str] = None
    error_message: Optional[str] = None