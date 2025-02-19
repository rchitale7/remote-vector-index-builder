# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import hashlib
from models.job import RequestParameters

def generate_job_id(request_parameters: RequestParameters) -> str:
    combined = str(request_parameters).encode()
    return hashlib.sha256(combined).hexdigest()