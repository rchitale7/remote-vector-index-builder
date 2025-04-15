# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from fastapi import APIRouter, Request
import json

import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/_jobs")
def get_jobs(request: Request) -> str:
    job_service = request.app.state.job_service
    jobs = job_service.get_jobs()
    return json.dumps(jobs, default=lambda o: o.__dict__, indent=4)