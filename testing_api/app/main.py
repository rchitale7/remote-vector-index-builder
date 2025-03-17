# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from app.api.routes import build, status
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.base.config import Settings
from app.base.resources import ResourceManager
from app.executors.workflow_executor import WorkflowExecutor
from app.services.index_builder import IndexBuilder
from app.services.job_service import JobService
from app.storage.factory import RequestStoreFactory
from app.utils.logging_config import configure_logging
from contextlib import asynccontextmanager

from app.utils.error_message import get_field_path

import logging

settings = Settings()

configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

request_store = RequestStoreFactory.create(
    store_type=settings.request_store_type,
    settings=settings
)

resource_manager = ResourceManager(
    total_gpu_memory=settings.gpu_memory_limit,
    total_cpu_memory=settings.cpu_memory_limit
)

index_builder = IndexBuilder(settings)

workflow_executor = WorkflowExecutor(
    max_workers=settings.max_workers,
    request_store=request_store,
    resource_manager=resource_manager,
    build_index_fn=index_builder.build_index
)

job_service = JobService(
    request_store=request_store,
    resource_manager=resource_manager,
    workflow_executor=workflow_executor,
    total_gpu_memory=settings.gpu_memory_limit,
    total_cpu_memory=settings.cpu_memory_limit
)

app = FastAPI(
    title=settings.service_name
)

app.state.job_service = job_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Shutting down application ...")
    workflow_executor.shutdown()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field_path = get_field_path(error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })

    logger.info(f"Error while validating parameters: #{errors}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "errors": errors
        }
    )

app.include_router(build.router)
app.include_router(status.router)