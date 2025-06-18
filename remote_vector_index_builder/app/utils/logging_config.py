# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging


def configure_logging(log_level):
    root_logger = (
        logging.getLogger()
    )  # root logging defaults to WARN, so setting as INFO here
    root_logger.setLevel(logging.INFO)

    logger = logging.getLogger("remote_vector_index_builder")

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    logger.addHandler(handler)
    logger.setLevel(level=log_level)
