# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic_settings import BaseSettings
from storage.types import RequestStoreType
from typing import Optional
from repositories.types import BlobStoreType

class Settings(BaseSettings):

    """
    Settings class for the application. Pulls the settings
    from the Docker container environment variables
    """

    blob_container_type: BlobStoreType = BlobStoreType.S3

    # s3 container settings
    s3_role_arn: Optional[str] = None
    s3_region: str = 'us-west-2'
    s3_retries: int = 3

    # Request Store settings
    request_store_type: RequestStoreType

    # In memory settings
    request_store_max_size: int = 10
    request_store_ttl_seconds: Optional[int] = None

    # Resource Manager settings
    gpu_memory_limit: float
    cpu_memory_limit: float

    # Workflow Executor settings
    max_workers: int = 5

    # Service settings
    service_name: str
    log_level: str

