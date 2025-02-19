# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.config import Settings
from enum import Enum
from storage.base import RequestStore
from storage.memory import InMemoryRequestStore
from storage.types import RequestStoreType

class RequestStoreFactory:
    @staticmethod
    def create(store_type: RequestStoreType, settings: Settings) -> RequestStore:
        if store_type == RequestStoreType.MEMORY:
            return InMemoryRequestStore(settings)
        else:
            raise ValueError(f"Unsupported request store type: {store_type}")