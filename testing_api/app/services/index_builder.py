# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
from typing import Optional, Tuple
import tempfile
import time
from app.models.workflow import BuildWorkflow
from core import run_tasks

logger = logging.getLogger(__name__)

# TODO: Implement object store, GPU builder clients
class IndexBuilder:
    def __init__(self, settings):
        self.settings = settings

    def build_index(self, workflow: BuildWorkflow) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Builds the index for the given workflow.
        Returns (success, index_path, message).
        """
        result = run_tasks(workflow.index_build_parameters)
        if not result.remote_path:
            logger.info("Failed to build index!")
            return False, None, result.error
        logger.info("Index built successfully!")
        logger.info(f"Index path: {result.remote_path}")
        return True, result.remote_path, "success!"
