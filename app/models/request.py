# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from pydantic import BaseModel
from typing import Optional

class RequestParameters(BaseModel):
    object_path: str
    tenant_id: str = ""

    def __str__(self):
        return f"{self.object_path}-{self.tenant_id}"

    def __eq__(self, other):
        if not isinstance(other, RequestParameters):
            return False
        return str(self) == str(other)