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


def configure_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():

    knn_vec = os.environ.get('KNN_VEC')
    knn_did = os.environ.get('KNN_DID')
    dimension = os.environ.get('DIMENSION')
    doc_count = os.environ.get('DOC_COUNT')

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
    run_tasks(model)
    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    configure_logging("INFO")
    main()
