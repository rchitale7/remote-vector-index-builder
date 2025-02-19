# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
ENV PYTHONPATH=/code/app

ENV REQUEST_STORE_TYPE='memory'

ENV GPU_MEMORY_LIMIT=24.0
ENV CPU_MEMORY_LIMIT=32.0

ENV SERVICE_NAME='remote-vector-index-builder'

ENV LOG_LEVEL="INFO"

CMD ["fastapi", "run", "app/main.py", "--port", "80"]