# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod

class BlobStore(ABC):

    @abstractmethod
    def read_blob(self, path: str, temp_dir: str) -> str:
        pass

    @abstractmethod
    def write_blob(self, path: str, temp_dir: str) -> str:
        pass

