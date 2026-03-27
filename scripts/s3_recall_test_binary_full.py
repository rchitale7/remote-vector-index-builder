#!/usr/bin/env python3
"""Recall test for binary HNSW indices that already have BQ vectors as storage.

Downloads a binary .faiss index from S3, quantizes query vectors using one-bit
scalar quantization, and measures recall with hamming distance search.
"""

import os
import sys
import logging
import argparse
import boto3
import tempfile
import numpy as np
import faiss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.dataset.dataset_utils import prepare_search_dataset, prepare_indexing_dataset
from benchmarking.utils.common_utils import recall_at_r
from timeit import default_timer as timer
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_from_s3(bucket, key, local_path, region='us-east-1'):
    s3 = boto3.client('s3', region_name=region)
    s3.download_file(bucket, key, local_path)
    logging.info(f"Downloaded s3://{bucket}/{key} to {local_path}")


def main():
    parser = argparse.ArgumentParser(description='Recall test for binary HNSW index with BQ storage')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--index-s3-key', required=True, help='S3 key for binary .faiss index')
    parser.add_argument('--dataset', required=True, help='HDF5 dataset (for queries, ground truth, and quantization thresholds)')
    parser.add_argument('--ef-search', type=int, default=100)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--region', default='us-east-1')

    args = parser.parse_args()
    setup_logging()

    # Download index
    with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as f:
        temp_index_path = f.name
    download_from_s3(args.bucket, args.index_s3_key, temp_index_path, args.region)

    # Load binary index
    index = faiss.read_index_binary(temp_index_path)
    hnsw_index = faiss.downcast_IndexBinary(index.index) if isinstance(index, faiss.IndexBinaryIDMap) else index
    hnsw_index.hnsw.efSearch = args.ef_search
    logging.info(f"Loaded binary index, ntotal={index.ntotal}")
    os.unlink(temp_index_path)

    # Compute quantization thresholds from training vectors
    d_idx, train_vectors, _ = prepare_indexing_dataset(args.dataset, args.normalize)
    thresholds = train_vectors.mean(axis=0)
    del train_vectors

    # Load queries and ground truth
    d, query_vectors, ground_truth = prepare_search_dataset(args.dataset, args.normalize)

    # Quantize queries with same thresholds
    query_bq = np.packbits((query_vectors > thresholds).astype(np.uint8), axis=1)
    logging.info(f"Quantized {len(query_vectors)} query vectors")

    # Search
    total_time = 0
    I = []
    for query in tqdm(query_bq, desc=f"Searching (ef_search={args.ef_search})"):
        t1 = timer()
        D, ids = index.search(np.array([query]), args.k)
        t2 = timer()
        I.append(ids[0])
        total_time += (t2 - t1)

    recall_k = recall_at_r(I, ground_truth, args.k, args.k, len(query_vectors))
    recall_1 = recall_at_r(I, ground_truth, 1, 1, len(query_vectors))

    logging.info(f"Recall@{args.k}: {recall_k}")
    logging.info(f"Recall@1: {recall_1}")
    logging.info(f"Search time: {total_time:.2f}s")
    logging.info(f"Throughput: {len(query_vectors) / total_time:.1f} queries/s")


if __name__ == "__main__":
    main()
