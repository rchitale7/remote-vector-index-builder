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


def main(
        knn_vec,
        knn_did,
        dimension,
        doc_count
):

    download = os.environ.get("DOWNLOAD", False)
    build = os.environ.get("BUILD", False)
    upload = os.environ.get("UPLOAD", False)
    multipart_chunksizes = os.environ.get("MULTIPART_CHUNKSIZE", None)
    thread_counts = os.environ.get("THREAD_COUNTS", None)


    tag = ""
    run_options = {
        "download": download,
        "build": build,
        "upload": upload,
        "local_path": f"/home/rchital/files/{knn_vec}"
    }
    if download:
        tag += "download"
    if build:
        tag += "_build"
    if upload:
        tag += "_upload"

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
    raw_data = []

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

            raw_data.append([i, chunk] + times)

            logging.info(f"ThreadCount: {i}, ChunkSize: {chunk}, P50: {p50}, P90: {p90}, Range: {data_range}")
    data = [['ThreadCount', 'ChunkSize', 'P50', 'P99', 'Range']] + data
    raw_data = [['ThreadCount', 'ChunkSize', 'Run_1', 'Run_2', 'Run_3', 'Run_4', 'Run_5', 'Run_6', 'Run_7', 'Run_8', 'Run_9', 'Run_10']] + raw_data

    cur_time = f"{datetime.now()}".replace(" ", "_")

    local_path = os.path.join('/tmp', f'{knn_vec}_{tag}_{cur_time}.csv')
    with open(local_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    s3_client = S3ObjectStore(model, {})
    s3_client.write_blob(local_path, f'{knn_vec}_{tag}_{cur_time}.csv')

    local_path_raw = os.path.join('/tmp', f'{knn_vec}_{tag}_{cur_time}_raw.csv')
    with open(local_path_raw, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(raw_data)

    s3_client = S3ObjectStore(model, {})
    s3_client.write_blob(local_path_raw, f'{knn_vec}_{tag}_{cur_time}_raw.csv')


if __name__ == "__main__":
    configure_logging("INFO")
    logger.info(f"Starting service!")

    logger.info(f"Running for 92 MB file")
    knn_vec = "-v7uZ5UBxsakWdsU-Go9_target_field__3l.knnvec"
    knn_did = "-v7uZ5UBxsakWdsU-Go9_target_field__3l.knndid"
    dimension = 768
    doc_count  = 33162


    main(
        knn_vec=knn_vec,
        knn_did=knn_did,
        dimension=dimension,
        doc_count=doc_count

    )

    logger.info(f"Running for 2.7 GB file")
    knn_vec = "DLVt1tJzSHmBjJlkz1KAkA_target_field__kw.knnvec"
    knn_did = "DLVt1tJzSHmBjJlkz1KAkA_target_field__kw.knndid"
    dimension = 768
    doc_count = 1000000



    main(
        knn_vec=knn_vec,
        knn_did=knn_did,
        dimension=dimension,
        doc_count=doc_count
    )

    logger.info(f"Running for a 27 GB file")
    knn_vec = "Qy-pPvYST8iBbK6DqWwdBg_target_field__427.knnvec"
    knn_did = "Qy-pPvYST8iBbK6DqWwdBg_target_field__427.knndid"
    dimension = 768
    doc_count = 10000000


    main(
        knn_vec=knn_vec,
        knn_did=knn_did,
        dimension=dimension,
        doc_count=doc_count
    )