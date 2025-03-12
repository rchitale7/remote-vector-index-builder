# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from fastapi import APIRouter, HTTPException, Request
from app.schemas.api import GetStatusResponse

router = APIRouter()

@router.get("/_status/{job_id}")
def get_status(job_id: str, request: Request) -> GetStatusResponse:

    job_service = request.app.state.job_service
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response_data = {
        "task_status": job.status
    }

    if hasattr(job, 'file_name')  and job.file_name is not None:
        response_data["file_name"] = job.file_name

    if hasattr(job, 'error_message') and job.error_message is not None:
        response_data["error_message"] = job.error_message

    return GetStatusResponse(**response_data)