# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

FROM rchitale7/remote-index-build-service:core

WORKDIR /testing_api

COPY app/requirements.txt /testing_api/app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /testing_api/app/requirements.txt

COPY . /testing_api

ENV PYTHONPATH='${PYTHONPATH}:/tmp/faiss/build/faiss/python:/remote_vector_index_builder:/testing_api'

ENV REQUEST_STORE_TYPE='memory'

ENV GPU_MEMORY_LIMIT=24
ENV CPU_MEMORY_LIMIT=64

ENV SERVICE_NAME='remote-vector-index-builder'

ENV LOG_LEVEL="INFO"

RUN ["python", "test_imports.py"]
CMD ["fastapi", "run", "app/main.py", "--port", "80"]