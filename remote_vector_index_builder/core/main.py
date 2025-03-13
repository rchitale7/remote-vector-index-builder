# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models.index_build_parameters import IndexBuildParameters
import time
from da
from core import create_vectors_dataset, upload_index, run_tasks
import logging
from io import BytesIO
import os
import csv
import numpy as np
from datetime import datetime

from core.object_store.s3.s3_object_store import S3ObjectStore
logger = logging.getLogger(__name__)


def configure_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():

    logger.info(f"Starting service!")
    knn_vec = os.environ.get('KNN_VEC')
    knn_did = os.environ.get('KNN_DID')
    dimension = os.environ.get('DIMENSION')
    doc_count = os.environ.get('DOC_COUNT')
    download = os.environ.get("DOWNLOAD", False)
    build = os.environ.get("BUILD", False)
    upload = os.environ.get("UPLOAD", False)
    local_path = os.environ.get("LOCAL_PATH", None)
    multipart_chunksizes = os.environ.get("MULTIPART_CHUNKSIZE", None)
    thread_counts = os.environ.get("THREAD_COUNTS", None)


    run_options = {
        "download": download,
        "build": build,
        "upload": upload,
        "local_path": local_path
    }

    index_build_params = {
        "repository_type": "s3",
        "container_name": "testbucket-rchital",
        "vector_path": knn_vec,
        "doc_id_path": knn_did,
        "dimension": dimension,
        "doc_count": doc_count,
    }

    model = IndexBuildParameters.model_validate(index_build_params)

    chunks = list(map(int, multipart_chunksizes.split(",")))
    thread_counts = list(map(int, thread_counts.split(",")))

    data = []

    for i in thread_counts:
        for chunk in chunks:
            object_store_config = {
                "transfer_config": {

                }
            }
            mb_chunk = chunk * 1024 * 1024
            object_store_config["transfer_config"]["multipart_chunksize"] = mb_chunk
            object_store_config["transfer_config"]["max_concurrency"] = i
            times = []
            for j in range(0,10):
                start_time = time.time()
                run_tasks(model, object_store_config, run_options)
                end_time = time.time()
                times.append(end_time - start_time)
            p50 = np.percentile(times, 50)
            p90 = np.percentile(times, 90)
            data_range = np.ptp(times)

            data.append(
                [i, chunk, p50, p90, data_range]
            )
            logging.info(f"ThreadCount: {i}, ChunkSize: {chunk}, TotalTime: {avg_time}")
    data = sorted(data, key=lambda x: x[-1])
    data = [['ThreadCount', 'ChunkSize', 'P50', 'P99', 'Range']] + data

    cur_time = f"{datetime.now()}".replace(" ", "_")
    local_path = os.path.join('/tmp', f'{knn_vec}_{cur_time}_output.csv')
    with open(local_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    s3_client = S3ObjectStore(model, {})
    s3_client.write_blob(local_path, f'{knn_vec}_upload_output.csv' )


if __name__ == "__main__":
    configure_logging("INFO")
    main()
