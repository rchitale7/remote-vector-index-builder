# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from app.base.exceptions import HashCollisionError, CapacityError
from fastapi import APIRouter, HTTPException, Request
from app.schemas.api import CreateJobResponse
from core.common.models.index_build_parameters import IndexBuildParameters

import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/_build")
def create_job(index_build_parameters: IndexBuildParameters, request: Request) -> CreateJobResponse:

    logger.info(f"Received build request: {index_build_parameters}")

    try:
        job_service = request.app.state.job_service
        job_id = job_service.create_job(index_build_parameters)
    except HashCollisionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except CapacityError as e:
        raise HTTPException(status_code=507, detail=str(e))
    return CreateJobResponse(job_id=job_id)