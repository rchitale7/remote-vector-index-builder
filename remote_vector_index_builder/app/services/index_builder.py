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
        storage_mode = os.environ.get("STORAGE_MODE", "disk")
        result = run_tasks(
            workflow.index_build_parameters,
            {"S3_ENDPOINT_URL": s3_endpoint_url},
            storage_mode,
        )
        if not result.file_name:
            return False, None, result.error
        return True, result.file_name, None
