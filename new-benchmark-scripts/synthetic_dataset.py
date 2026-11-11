#!/usr/bin/env python3
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic vector dataset and upload it to S3 as .knnvec and _ids.knndid.

Use this to benchmark build time and memory at a shape no public dataset has (e.g. 8.33M x 1024).
There is no ground truth, so recall can't be measured on these datasets.

Vectors are drawn from a Gaussian mixture (random cluster centers plus per-vector noise) so that
IVF clustering sees structure similar to real embeddings; uniform noise would make every list the
same size and every neighbor equally far away. For innerproduct and cosinesimil the vectors are
L2-normalized, like typical embedding models produce.

The upload is streamed in parts, so host memory stays at one chunk regardless of the dataset size.

Usage:
    python synthetic_dataset.py --name synthetic-8330000-1024-innerproduct --doc-count 8330000 --dimension 1024 \
        --space-type innerproduct --bucket my-bucket --s3-prefix datasets/float --dtype float32
"""

import argparse
import logging

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# S3 multipart parts must be at least 5 MiB (except the last); ~256 MiB parts keep the part count low.
PART_BYTES = 256 * 1024 * 1024


def multipart_upload(s3, bucket, key, chunks):
    """Upload an iterable of byte chunks as one S3 object, buffering up to PART_BYTES per part."""
    upload_id = s3.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    parts, buf, total = [], bytearray(), 0
    try:
        def flush():
            n = len(parts) + 1
            etag = s3.upload_part(Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=n, Body=bytes(buf))["ETag"]
            parts.append({"PartNumber": n, "ETag": etag})
            buf.clear()

        for chunk in chunks:
            buf += chunk
            total += len(chunk)
            if len(buf) >= PART_BYTES:
                flush()
        if buf or not parts:
            flush()
        s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
    except Exception:
        s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise
    logging.info(f"  Uploaded {total / 1e9:.2f} GB to s3://{bucket}/{key} in {len(parts)} parts")


def vector_chunks(doc_count, dimension, space_type, dtype, clusters, noise, seed, chunk_rows):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dimension), dtype=np.float32)
    normalize = space_type in ("innerproduct", "cosinesimil")
    for start in range(0, doc_count, chunk_rows):
        rows = min(chunk_rows, doc_count - start)
        v = centers[rng.integers(0, clusters, rows)]
        v += noise * rng.standard_normal((rows, dimension), dtype=np.float32)
        if normalize:
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        yield v.astype("<f2" if dtype == "float16" else "<f4").tobytes()
        if (start // chunk_rows) % 10 == 0:
            logging.info(f"  generated {start + rows:,} / {doc_count:,} vectors")


def main():
    p = argparse.ArgumentParser(description="Generate a synthetic dataset and upload it as .knnvec/_ids.knndid")
    p.add_argument("--name", required=True, help="Dataset name; objects are <s3-prefix>/<name>.knnvec and _ids.knndid")
    p.add_argument("--doc-count", type=int, required=True)
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--space-type", default="l2", choices=["l2", "innerproduct", "cosinesimil"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"], help="Encoding of the uploaded .knnvec")
    p.add_argument("--clusters", type=int, default=1024, help="Gaussian mixture components")
    p.add_argument("--noise", type=float, default=0.5, help="Per-vector noise relative to the unit-variance centers")
    p.add_argument("--seed", type=int, default=42, help="Same seed gives the same vectors for every --dtype")
    p.add_argument("--chunk-rows", type=int, default=100_000)
    p.add_argument("--bucket", required=True)
    p.add_argument("--s3-prefix", required=True, help="S3 key prefix (e.g. datasets/float)")
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    logging.info(f"Generating {args.name}: {args.doc_count:,} x {args.dimension} {args.dtype} ({args.space_type}) "
                 f"-> s3://{args.bucket}/{args.s3_prefix}/")
    multipart_upload(s3, args.bucket, f"{args.s3_prefix}/{args.name}.knnvec",
                     vector_chunks(args.doc_count, args.dimension, args.space_type, args.dtype,
                                   args.clusters, args.noise, args.seed, args.chunk_rows))
    ids = np.arange(args.doc_count, dtype="<i4").tobytes()
    multipart_upload(s3, args.bucket, f"{args.s3_prefix}/{args.name}_ids.knndid", [ids])
    logging.info("Done.")


if __name__ == "__main__":
    main()
