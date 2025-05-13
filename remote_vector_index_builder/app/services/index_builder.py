# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
import os
from typing import Optional, Tuple
from app.models.workflow import BuildWorkflow
from core.tasks import run_tasks
import sys

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Handles the building of indexes based on provided workflows.
    """

    def build_index(
        self, workflow: BuildWorkflow
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Builds the index for the given workflow, using the run_tasks function

        Args:
            workflow (BuildWorkflow): Workflow containing index build parameters

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: A tuple containing:
                - Success status (True/False)
                - Index path if successful, None otherwise
                - Error message if failed, None otherwise
        """
        s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)
        upload_io_chunksize = int(os.environ.get("UPLOAD_IO_CHUNKSIZE", sys.maxsize))
        upload_max_concurrency = int(os.environ.get("UPLOAD_MAX_CONCURRENCY", 2))
        upload_multipart_threshold = int(
            os.environ.get("UPLOAD_MULTIPART_THRESHOLD", 50 * 1024 * 1024)
        )

        upload_transfer_config = {
            "multipart_threshold": upload_multipart_threshold,
            "multipart_chunksize": upload_multipart_threshold,
            "max_concurrency": upload_max_concurrency,
            "io_chunksize": upload_io_chunksize
        }


        result = run_tasks(
            workflow.index_build_parameters, {
                "S3_ENDPOINT_URL": s3_endpoint_url,
                "upload_transfer_config": upload_transfer_config
            }
        )
        if not result.file_name:
            return False, None, result.error
        return True, result.file_name, None
