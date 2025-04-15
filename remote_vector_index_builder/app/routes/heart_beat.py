# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from fastapi import APIRouter

import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/_heart_beat")
def heart_beat() -> str:
    return "Hello!"