# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models.index_build_parameters import IndexBuildParameters
import time
from core import create_vectors_dataset, upload_index, run_tasks
import logging
from io import BytesIO
import os

logger = logging.getLogger(__name__)


def configure_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():

    logger.info(f"Starting service!")
    logger.info(f"OS CPU count: {os.cpu_count()}")
    knn_vec = os.environ.get('KNN_VEC')
    knn_did = os.environ.get('KNN_DID')
    dimension = os.environ.get('DIMENSION')
    doc_count = os.environ.get('DOC_COUNT')
    download = os.environ.get("DOWNLOAD", False)
    build = os.environ.get("BUILD", False)
    upload = os.environ.get("UPLOAD", False)
    local_path = os.environ.get("LOCAL_PATH", None)
    multipart_threshold = os.environ.get("MULTIPART_THRESHOLD", None)
    multipart_chunksize = os.environ.get("MULTIPART_CHUNKSIZE", None)
    max_concurrency_factor = os.environ.get("MAX_CONCURRENCY_FACTOR", None)
    checksum_mode = os.environ.get("CHECKSUM_MODE", None)
    checksum_algorithm = os.environ.get("CHECKSUM_ALGORITHM", None)


    run_options = {
        "download": download,
        "build": build,
        "upload": upload,
        "local_path": local_path
    }

    object_store_config = {
        'transfer_config': {

        },
        'download_args': {

        },
        'upload_args': {

        }
    }
    if multipart_threshold is not None:
        object_store_config["transfer_config"]["multipart_threshold"] = int(multipart_threshold)
    if multipart_chunksize is not None:
        object_store_config["transfer_config"]["multipart_chunksize"] = int(multipart_chunksize)
    if max_concurrency_factor is not None:
        object_store_config["transfer_config"]["max_concurrency"] = os.cpu_count() // int(max_concurrency_factor)
    if checksum_mode is not None:
        object_store_config["transfer_config"]["download_args"]["ChecksumMode"] = checksum_mode
    if checksum_algorithm is not None:
        object_store_config["transfer_config"]["upload_args"]["ChecksumAlgorithm"] = checksum_algorithm


    index_build_params = {
        "repository_type": "s3",
        "container_name": "testbucket-rchital",
        "vector_path": knn_vec,
        "doc_id_path": knn_did,
        "dimension": dimension,
        "doc_count": doc_count,
    }

    model = IndexBuildParameters.model_validate(index_build_params)
    start_time = time.time()
    run_tasks(model, object_store_config, run_options)
    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    configure_logging("INFO")
    main()
