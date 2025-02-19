# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from schemas.api import CreateJobRequest
from models.request import RequestParameters

def create_request_parameters(create_job_request: CreateJobRequest) -> RequestParameters:
    return RequestParameters(
        object_path=create_job_request.object_path,
        tenant_id=create_job_request.tenant_id
    )