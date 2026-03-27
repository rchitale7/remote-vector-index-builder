#!/usr/bin/env python3
"""Recall test for graph-only indices with BQ (binary quantized) vectors.

Takes an fp32 graph-only .faiss index and BQ vectors from S3, copies the HNSW
graph into an IndexBinaryHNSW, attaches the BQ vectors as storage, quantizes
query vectors using one-bit scalar quantization (per-dimension mean thresholds
from training data), and measures recall with hamming distance search.
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


def build_binary_index_from_graph(graph_index, bq_vectors, dimension, M):
    """Copy HNSW graph from fp32 index into an IndexBinaryHNSW with BQ storage."""
    # Unwrap IndexIDMap if present
    inner = faiss.downcast_index(graph_index.index) if isinstance(graph_index, faiss.IndexIDMap) else graph_index
    binary_index = faiss.IndexBinaryHNSW(dimension, M)

    src_hnsw = inner.hnsw
    dst_hnsw = binary_index.hnsw

    src_neighbors = faiss.vector_to_array(src_hnsw.neighbors)
    src_offsets = faiss.vector_to_array(src_hnsw.offsets)
    src_levels = faiss.vector_to_array(src_hnsw.levels)

    faiss.copy_array_to_vector(src_neighbors, dst_hnsw.neighbors)
    faiss.copy_array_to_vector(src_offsets, dst_hnsw.offsets)
    faiss.copy_array_to_vector(src_levels, dst_hnsw.levels)

    # Recover entry_point and max_level if they were lost during graph-only serialization
    if src_hnsw.entry_point >= 0:
        dst_hnsw.entry_point = src_hnsw.entry_point
        dst_hnsw.max_level = src_hnsw.max_level
    else:
        max_level_val = int(src_levels.max())
        dst_hnsw.max_level = max_level_val - 1
        dst_hnsw.entry_point = int(np.where(src_levels == max_level_val)[0][0])
        logging.info(f"Recovered entry_point={dst_hnsw.entry_point}, max_level={dst_hnsw.max_level}")

    dst_hnsw.efSearch = src_hnsw.efSearch
    dst_hnsw.efConstruction = src_hnsw.efConstruction

    binary_index.ntotal = len(bq_vectors)
    binary_index.storage.add(bq_vectors)

    return binary_index


def main():
    parser = argparse.ArgumentParser(description='Recall test for graph-only + BQ vectors')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--index-s3-key', required=True, help='S3 key for fp32 graph-only .faiss index')
    parser.add_argument('--vectors-s3-key', required=True, help='S3 key for BQ .knnvec file (packed binary)')
    parser.add_argument('--dataset', required=True, help='HDF5 dataset (for queries, ground truth, and quantization thresholds)')
    parser.add_argument('--dimension', type=int, required=True, help='Original float dimension (= number of bits in BQ)')
    parser.add_argument('--m', type=int, default=16, help='HNSW M parameter')
    parser.add_argument('--ef-search', type=int, default=100)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--region', default='us-east-1')

    args = parser.parse_args()
    setup_logging()

    # Download graph-only index
    with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as f:
        temp_index_path = f.name
    download_from_s3(args.bucket, args.index_s3_key, temp_index_path, args.region)

    # Download BQ vectors
    with tempfile.NamedTemporaryFile(delete=False, suffix='.knnvec') as f:
        temp_vec_path = f.name
    download_from_s3(args.bucket, args.vectors_s3_key, temp_vec_path, args.region)

    # Load graph-only index
    graph_index = faiss.read_index(temp_index_path)
    logging.info(f"Loaded graph-only index, ntotal={graph_index.ntotal}")
    os.unlink(temp_index_path)

    # Load BQ vectors
    bytes_per_vector = args.dimension // 8
    bq_vectors = np.fromfile(temp_vec_path, dtype=np.uint8).reshape(-1, bytes_per_vector)
    logging.info(f"Loaded {len(bq_vectors)} BQ vectors ({bytes_per_vector} bytes each)")
    os.unlink(temp_vec_path)

    # Build binary HNSW with copied graph + BQ storage
    binary_index = build_binary_index_from_graph(graph_index, bq_vectors, args.dimension, args.m)
    binary_index.hnsw.efSearch = args.ef_search
    del graph_index, bq_vectors
    logging.info("Built IndexBinaryHNSW with copied graph and BQ storage")

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
        D, ids = binary_index.search(np.array([query]), args.k)
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
