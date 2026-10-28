#!/usr/bin/env python3
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convert local HDF5 datasets to .knnvec and _ids.knndid and upload to S3.
Reads dataset list from a CSV file.

Usage:
    python dataset_to_local.py --csv datasets.csv --input-dir ../../datasets --bucket testbucket-rchital --s3-prefix datasets/float
    python dataset_to_local.py ... --s3-prefix datasets/half_float --dtype float16
    python dataset_to_local.py ... --s3-prefix datasets/byte --dtype byte
    python dataset_to_local.py ... --s3-prefix datasets/binary --dtype binary   (same as --quantize)
"""

import os
import sys
import csv
import logging
import argparse
import numpy as np
import boto3
import json
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarking.dataset.dataset_utils import prepare_indexing_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def quantize_to_binary(vectors: np.ndarray) -> bytes:
    """One-bit scalar quantization matching k-NN OneBitScalarQuantizer + BitPacker."""
    thresholds = vectors.mean(axis=0)
    bits = (vectors > thresholds).astype(np.uint8)
    return np.packbits(bits, axis=1).tobytes()


def quantize_to_byte(vectors: np.ndarray):
    """Scale vectors into the int8 range [-128, 127], like a k-NN byte field.

    Returns the int8 bytes and the scale, so queries can be scaled the same way at search time.
    """
    scale = 127.0 / float(np.abs(vectors).max())
    return np.clip(np.rint(vectors * scale), -128, 127).astype("<i1").tobytes(), scale


def upload_to_s3(data: bytes, bucket: str, key: str, s3_client):
    s3_client.upload_fileobj(BytesIO(data), bucket, key)
    logging.info(f"  Uploaded {len(data)/1e6:.1f}MB to s3://{bucket}/{key}")


def convert_and_upload(hdf5_path: str, bucket: str, s3_prefix: str, s3_client,
                       normalize: bool, dtype: str):
    basename = os.path.splitext(os.path.basename(hdf5_path))[0]
    logging.info(f"Processing {basename}...")

    d, vectors, ids = prepare_indexing_dataset(hdf5_path, normalize, -1)
    ids_array = np.array(ids, dtype=np.int32)

    vec_key = f"{s3_prefix}/{basename}.knnvec"
    ids_key = f"{s3_prefix}/{basename}_ids.knndid"

    if dtype == "binary":
        logging.info("  Applying one-bit scalar quantization...")
        vec_bytes = quantize_to_binary(vectors)
    elif dtype == "float16":
        # k-NN half_float fields upload fp16 (2 bytes per value)
        vec_bytes = vectors.astype("<f2").tobytes()
    elif dtype == "byte":
        vec_bytes, scale = quantize_to_byte(vectors)
        # recall_test_csv.py reads this to scale queries into the same int8 space
        upload_to_s3(json.dumps({"scale": scale}).encode(), bucket,
                     f"{s3_prefix}/{basename}_byte_scale.json", s3_client)
    else:
        vec_bytes = vectors.astype("<f4").tobytes()

    upload_to_s3(vec_bytes, bucket, vec_key, s3_client)
    upload_to_s3(ids_array.tobytes(), bucket, ids_key, s3_client)

    logging.info(f"  {basename}: {len(vectors)} vectors, dim={d}")


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 datasets to .knnvec/_ids.knndid and upload to S3')
    parser.add_argument('--csv', required=True, help='CSV file listing datasets')
    parser.add_argument('--input-dir', required=True, help='Directory containing HDF5 files')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-prefix', required=True, help='S3 key prefix (e.g. datasets/float)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'byte', 'binary'],
                        help='Encoding of the uploaded .knnvec')
    parser.add_argument('--quantize', action='store_true', help='Apply one-bit scalar quantization (same as --dtype binary)')
    args = parser.parse_args()
    if args.quantize:
        args.dtype = 'binary'

    s3_client = boto3.client('s3', region_name=args.region)

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    logging.info(f"Converting {len(rows)} dataset(s) ({args.dtype}) -> s3://{args.bucket}/{args.s3_prefix}/")
    for row in rows:
        hdf5_path = os.path.join(args.input_dir, row['filename'])
        normalize = row['normalize'].lower() == 'true'
        convert_and_upload(hdf5_path, args.bucket, args.s3_prefix, s3_client, normalize, args.dtype)

    logging.info("Done.")


if __name__ == "__main__":
    main()
