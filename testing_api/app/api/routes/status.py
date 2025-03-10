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

    if hasattr(job, 'file_path')  and job.file_path is not None:
        response_data["file_path"] = job.file_path

    if hasattr(job, 'msg') and job.msg is not None:
        response_data["msg"] = job.msg

    return GetStatusResponse(**response_data)