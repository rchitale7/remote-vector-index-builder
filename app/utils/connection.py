# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from functools import cache
from botocore.config import Config

import boto3

import logging

logger = logging.getLogger(__name__)

@cache
def get_boto3_client(s3_role_arn: str, region: str, s3_retries: int):
    logger.info("Creating boto3 client for region %s", region)
    config = Config(retries={"max_attempts": s3_retries})
    return boto3.client("s3", region_name=region, config=config)