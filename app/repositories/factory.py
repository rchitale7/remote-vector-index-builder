# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from repositories.s3_blob_store import S3BlobStore
from repositories.blob_store import BlobStore
from repositories.types import BlobStoreType
from core.config import Settings
from schemas.api import CreateJobRequest

class BlobStoreFactory:
    @staticmethod
    def create(create_job_request: CreateJobRequest, settings: Settings) -> BlobStore:
        if create_job_request.repository_type == BlobStoreType.S3:
            return S3BlobStore(create_job_request.container_name, settings)
        else:
            raise ValueError(f"Unsupported blob store type: {create_job_request.store_type}")