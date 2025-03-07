# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models.index_build_parameters import IndexBuildParameters
from app.models.request import RequestParameters

def create_request_parameters(index_build_parameters: IndexBuildParameters) -> RequestParameters:
    return RequestParameters(
        vector_path=index_build_parameters.vector_path,
        tenant_id=index_build_parameters.tenant_id
    )