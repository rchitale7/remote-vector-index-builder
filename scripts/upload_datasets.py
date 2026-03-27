#!/usr/bin/env python3
"""Upload benchmark datasets to S3 with optional --quantize flag for binary quantization.
Excludes cohere-10M.

Usage:
    python upload_datasets.py [--quantize]
"""

import argparse
import subprocess
import sys
import os

BUCKET = "testbucket-rchital"
REGION = "us-west-2"
LOCAL_DIR = "/home/ec2-user/k-nn/datasets"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_to_s3.py")

DATASETS = [
    {"name": "sift", "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5", "dim": 128, "doc_count": 1000000},
    {"name": "ms-marco-384", "url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/ms_marco-384-1m.hdf5?download=true", "dim": 384, "doc_count": 1000000},
    {"name": "cohere-768-l2", "url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/cohere-768-l2.hdf5?download=true", "dim": 768, "doc_count": 1000000},
    {"name": "cohere-768-ip", "url": "https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings/documents-1m.hdf5.bz2", "dim": 768, "doc_count": 1000000, "compressed": True},
    {"name": "gist", "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5", "dim": 960, "doc_count": 1000000},
    {"name": "open-ai-1536-temp", "url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/open-ai-1536-temp.hdf5?download=true", "dim": 1536, "doc_count": 1000000},
    {"name": "bigAnn-10M", "url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/bigann-10M-with-gt.hdf5?download=true", "dim": 128, "doc_count": 10000000},
]

parser = argparse.ArgumentParser()
parser.add_argument("--quantize", action="store_true", help="Apply one-bit scalar quantization")
args = parser.parse_args()

data_type = "binary" if args.quantize else "float"
os.makedirs(LOCAL_DIR, exist_ok=True)

for ds in DATASETS:
    folder = f"{ds['dim']}_{ds['doc_count']}"
    base_path = f"test-data/{data_type}/{folder}"

    cmd = [
        sys.executable, SCRIPT,
        "--download-url", ds["url"],
        "--dataset-name", ds["name"],
        "--bucket", BUCKET,
        "--vectors-key", f"{base_path}/{ds['name']}.knnvec",
        "--docids-key", f"{base_path}/{ds['name']}_ids.knndid",
        "--doc-count", "-1",
        "--region", REGION,
        "--local-dir", LOCAL_DIR,
    ]
    if ds.get("compressed"):
        cmd.append("--compressed")
    if args.quantize:
        cmd.append("--quantize")

    print(f"\n=== Uploading {ds['name']} ({ds['dim']}d, {ds['doc_count']} docs) ===")
    subprocess.run(cmd, check=True)
    print(f"Done: {ds['name']}")

print("\nAll datasets uploaded.")
