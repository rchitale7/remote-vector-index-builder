#!/usr/bin/env python3
"""Recall test that iterates S3 datasets, supporting float and binary graphs/vectors.

Loops through test-data/<graph-type>/ sub-folders, downloads graphs and vectors
(potentially from different data type folders), and measures recall.

Supported combinations:
  --graph-type float --vector-type float   : float HNSW with float storage
  --graph-type float --vector-type binary  : copies HNSW graph into IndexBinaryHNSW + BQ storage
  --graph-type binary --vector-type binary : loads IndexBinaryHNSW directly
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


def download_from_s3(s3, bucket, key, local_path):
    s3.download_file(bucket, key, local_path)
    logging.info(f"Downloaded s3://{bucket}/{key}")


def recover_hnsw_entry(hnsw):
    """Recover entry_point and max_level if lost during graph-only serialization."""
    if hnsw.entry_point >= 0:
        return
    levels = faiss.vector_to_array(hnsw.levels)
    max_level_val = int(levels.max())
    hnsw.max_level = max_level_val - 1
    hnsw.entry_point = int(np.where(levels == max_level_val)[0][0])
    logging.info(f"Recovered entry_point={hnsw.entry_point}, max_level={hnsw.max_level}")


def build_binary_from_float_graph(graph_index, bq_vectors, dimension, M):
    """Copy HNSW graph from fp32 index into IndexBinaryHNSW with BQ storage."""
    inner = faiss.downcast_index(graph_index.index) if isinstance(graph_index, faiss.IndexIDMap) else graph_index
    binary_index = faiss.IndexBinaryHNSW(dimension, M)
    src, dst = inner.hnsw, binary_index.hnsw

    faiss.copy_array_to_vector(faiss.vector_to_array(src.neighbors), dst.neighbors)
    faiss.copy_array_to_vector(faiss.vector_to_array(src.offsets), dst.offsets)
    faiss.copy_array_to_vector(faiss.vector_to_array(src.levels), dst.levels)
    dst.efSearch = src.efSearch
    dst.efConstruction = src.efConstruction

    if src.entry_point >= 0:
        dst.entry_point = src.entry_point
        dst.max_level = src.max_level
    else:
        recover_hnsw_entry(dst)

    binary_index.ntotal = len(bq_vectors)
    binary_index.storage.add(bq_vectors)
    return binary_index


def load_index(graph_path, vector_path, graph_type, vector_type, dimension, ef_search, M, is_ip=False):
    """Load graph + vectors, return (index, needs_quantization)."""
    if graph_type == "float" and vector_type == "float":
        index = faiss.read_index(graph_path)
        inner = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap) else index
        if inner.storage is None:
            vecs = np.fromfile(vector_path, dtype=np.float32).reshape(-1, dimension)
            flat = faiss.IndexFlatIP(dimension) if is_ip else faiss.IndexFlatL2(dimension)
            flat.add(vecs)
            inner.storage = flat
            inner.own_fields = True
            recover_hnsw_entry(inner.hnsw)
        inner.hnsw.efSearch = ef_search
        return index, False

    elif graph_type == "float" and vector_type == "binary":
        graph_index = faiss.read_index(graph_path)
        bq = np.fromfile(vector_path, dtype=np.uint8).reshape(-1, dimension // 8)
        bi = build_binary_from_float_graph(graph_index, bq, dimension, M)
        bi.hnsw.efSearch = ef_search
        return bi, True

    elif graph_type == "binary" and vector_type == "binary":
        index = faiss.read_index_binary(graph_path)
        hnsw = faiss.downcast_IndexBinary(index.index).hnsw if isinstance(index, faiss.IndexBinaryIDMap) else index.hnsw
        hnsw.efSearch = ef_search
        return index, True

    else:
        raise ValueError(f"Unsupported: graph_type={graph_type}, vector_type={vector_type}")


def find_dataset(datasets_dir, name):
    """Find HDF5 file matching graph name."""
    for f in os.listdir(datasets_dir):
        if f.endswith('.hdf5') and f.replace('.hdf5', '') == name:
            return os.path.join(datasets_dir, f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Recall test across S3 datasets')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--graph-type', default='float', choices=['float', 'binary'])
    parser.add_argument('--vector-type', default='float', choices=['float', 'binary'])
    parser.add_argument('--datasets-dir', required=True, help='Local dir with HDF5 files')
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--ef-search', type=int, default=256)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('-o', '--output', default='recall_results.csv', help='Output CSV file')

    args = parser.parse_args()
    setup_logging()

    s3 = boto3.client('s3', region_name=args.region)
    graph_prefix = f"test-data/{args.graph_type}/"
    vector_prefix = f"test-data/{args.vector_type}/"

    resp = s3.list_objects_v2(Bucket=args.bucket, Prefix=graph_prefix, Delimiter="/")
    sub_folders = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    results = []

    for folder in sub_folders:
        folder_name = folder.rstrip("/").split("/")[-1]
        if "_" not in folder_name:
            continue
        dimension = int(folder_name.split("_")[0])

        objs = s3.list_objects_v2(Bucket=args.bucket, Prefix=folder)
        graph_files = [o["Key"] for o in objs.get("Contents", []) if o["Key"].endswith(".faiss")]

        for graph_key in graph_files:
            name = graph_key.split("/")[-1].replace(".faiss", "")

            dataset_path = find_dataset(args.datasets_dir, name)
            if not dataset_path:
                logging.warning(f"No HDF5 for {name}, skipping")
                continue

            vector_key = f"{vector_prefix}{folder_name}/{name}.knnvec"
            is_ip = "-ip" in name

            logging.info(f"\n=== {name} (dim={dimension}, space={'ip' if is_ip else 'l2'}) ===")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as f:
                    tmp_graph = f.name
                download_from_s3(s3, args.bucket, graph_key, tmp_graph)

                tmp_vec = None
                needs_vectors = (args.graph_type != args.vector_type) or args.graph_type == "float"
                if needs_vectors:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.knnvec') as f:
                        tmp_vec = f.name
                    try:
                        download_from_s3(s3, args.bucket, vector_key, tmp_vec)
                    except Exception:
                        logging.warning(f"Vector file not found: {vector_key}, skipping")
                        os.unlink(tmp_graph)
                        continue

                index, needs_quant = load_index(
                    tmp_graph, tmp_vec, args.graph_type, args.vector_type,
                    dimension, args.ef_search, args.m, is_ip
                )
                os.unlink(tmp_graph)
                if tmp_vec and os.path.exists(tmp_vec):
                    os.unlink(tmp_vec)

                # Compute quantization thresholds if binary queries needed
                thresholds = None
                if needs_quant:
                    _, train_vecs, _ = prepare_indexing_dataset(dataset_path, False)
                    thresholds = train_vecs.mean(axis=0)
                    del train_vecs

                _, query_vecs, ground_truth = prepare_search_dataset(dataset_path, False)

                # Quantize queries if needed
                if needs_quant:
                    queries = np.packbits((query_vecs > thresholds).astype(np.uint8), axis=1)
                else:
                    queries = query_vecs

                # Search
                total_time = 0
                I = []
                for q in tqdm(queries, desc=f"Searching {name}"):
                    t1 = timer()
                    D, ids = index.search(np.array([q]), args.k)
                    t2 = timer()
                    I.append(ids[0])
                    total_time += (t2 - t1)

                recall_k = recall_at_r(I, ground_truth, args.k, args.k, len(queries))
                recall_1 = recall_at_r(I, ground_truth, 1, 1, len(queries))
                recall_100 = recall_at_r(I, ground_truth, 100, 100, len(queries)) if args.k >= 100 else None

                row = {
                    "dataset": name,
                    "dimension": dimension,
                    "graph_type": args.graph_type,
                    "vector_type": args.vector_type,
                    f"recall@{args.k}": round(recall_k, 6),
                    "recall@1": round(recall_1, 6),
                    "recall@100": round(recall_100, 6) if recall_100 is not None else "",
                    "search_time_s": round(total_time, 2),
                    "throughput_qps": round(len(queries) / total_time, 1),
                    "num_queries": len(queries),
                    "ef_search": args.ef_search,
                }
                results.append(row)
                logging.info(f"Recall@{args.k}: {recall_k:.4f}, Recall@1: {recall_1:.4f}, "
                             f"Throughput: {row['throughput_qps']} qps")
                del index

            except Exception as e:
                logging.error(f"Error processing {name}: {e}")
                continue

    print("\n=== Summary ===")
    for r in results:
        print(r)

    if results:
        import csv
        keys = results[0].keys()
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
