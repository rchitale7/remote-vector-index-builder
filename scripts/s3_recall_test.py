#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
import boto3
from io import BytesIO

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.search.search_indices import runIndicesSearch
from benchmarking.dataset.dataset_utils import prepare_search_dataset

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(levelname)s - %(message)s')

def download_from_s3(bucket: str, key: str, region: str = 'us-east-1') -> bytes:
    """Download binary data from S3"""
    s3_client = boto3.client('s3', region_name=region)
    buffer = BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    data = buffer.read()
    logging.info(f"Downloaded {len(data)} bytes from s3://{bucket}/{key}")
    return data

def binary_to_numpy(data: bytes, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Convert binary data back to numpy array"""
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def main():
    parser = argparse.ArgumentParser(description='Download binary file from S3 and test recall')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-key', required=True, help='S3 key for the binary file')
    parser.add_argument('--query-dataset', required=True, help='Path to query dataset (HDF5)')
    parser.add_argument('--index-file', required=True, help='Path to index file for search')
    parser.add_argument('--dimensions', type=int, required=True, help='Vector dimensions')
    parser.add_argument('--vector-count', type=int, required=True, help='Number of vectors in binary file')
    parser.add_argument('--ef-search', type=int, default=100, help='ef_search parameter')
    parser.add_argument('--k', type=int, default=100, help='K value for recall')
    parser.add_argument('--normalize', action='store_true', help='Normalize query vectors')
    parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()
    setup_logging()

    try:
        # Download binary data from S3
        binary_data = download_from_s3(args.bucket, args.s3_key, args.region)

        # Convert binary to numpy array
        vectors = binary_to_numpy(binary_data, (args.vector_count, args.dimensions))
        logging.info(f"Loaded {len(vectors)} vectors of dimension {args.dimensions}")

        # Prepare search dataset
        d, query_vectors, ground_truth = prepare_search_dataset(args.query_dataset, args.normalize)

        # Run search and calculate recall
        search_params = {
            'ef_search': args.ef_search,
            'K': args.k
        }

        results = runIndicesSearch(query_vectors, args.index_file, search_params, ground_truth)

        # Print results
        logging.info("Search Results:")
        for key, value in results.items():
            logging.info(f"{key}: {value}")

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
