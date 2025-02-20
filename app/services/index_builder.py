# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.config import Settings
from repositories.factory import BlobStoreFactory
from typing import Optional, Tuple
import tempfile
import time
from models.workflow import BuildWorkflow
from schemas.api import CreateJobRequest

import logging
logger = logging.getLogger(__name__)
import traceback


# TODO: Implement blob container, GPU builder clients
class IndexBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_index(self, workflow: BuildWorkflow) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Builds the index for the given workflow.
        Returns (success, index_path).
        """


        logger.info("Getting object")
        bloby_store = BlobStoreFactory.create(
            create_job_request=workflow.create_job_request,
            settings=self.settings
        )

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download vectors

            start_time = time.time()
            objects = bloby_store.read_blob(workflow.create_job_request.object_path, temp_dir)
            logger.info(f"Downloaded {objects}")
            # temp_file_path = bloby_container.download_normal(
            #     workflow.create_job_request.container_name,
            #     workflow.create_job_request.object_path,
            #     temp_dir
            # )

            end_time = time.time()
            logger.info(f"Time taken: {end_time - start_time} seconds")

            vector_path = self._download_vectors(
                workflow.create_job_request,
                temp_dir
            )

            # Build index
            index_path = self._build_gpu_index(
                vector_path,
                workflow.create_job_request,
                temp_dir
            )

            # Upload index
            final_path = self._upload_index(
                index_path,
                workflow.create_job_request,
                temp_dir
            )

            return True, final_path, "success!"

    def _download_vectors(self, create_job_request: CreateJobRequest, temp_dir: str) -> str:
        """
        Download vectors from object store to temporary directory.
        Returns local path to vectors file.
        TODO: use object store client from object_store package
        """
        time.sleep(5)
        return "done"

    def _build_gpu_index(self, vector_path: str, create_job_request: CreateJobRequest, temp_dir: str) -> str:
        """
        Build GPU index
        Returns path to built index.
        TODO: use builder client from builder package
        """
        time.sleep(5)
        return "done"

    def _upload_index(self, index_path: str, create_job_request: CreateJobRequest, temp_dir: str) -> str:
        """
        Upload built index to object store.
        Returns final object store path.
        TODO: use object store client from object_store package
        """
        time.sleep(5)
        return "done"
