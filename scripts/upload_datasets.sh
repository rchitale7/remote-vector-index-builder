#!/usr/bin/env bash
# Upload benchmark datasets to S3 with optional --quantize flag for binary quantization.
# Excludes cohere-10M.
# Usage: ./upload_datasets.sh [--quantize]

set -e

BUCKET="testbucket-rchital"
REGION="us-west-2"
LOCAL_DIR="/home/ec2-user/k-nn/datasets"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/dataset_to_s3.py"

QUANTIZE_FLAG=""
DATA_TYPE="float"
if [[ "$1" == "--quantize" ]]; then
    QUANTIZE_FLAG="--quantize"
    DATA_TYPE="binary"
fi

BASE_PATH="test-data/${DATA_TYPE}"

declare -A DATASETS=(
    ["sift-128"]="https://ann-benchmarks.com/sift-128-euclidean.hdf5|sift|128|1000000"
    ["ms-marco-384"]="https://huggingface.co/datasets/navneet1v/datasets/resolve/main/ms_marco-384-1m.hdf5?download=true|ms-marco-384|384|1000000"
    ["cohere-768-l2"]="https://huggingface.co/datasets/navneet1v/datasets/resolve/main/cohere-768-l2.hdf5?download=true|cohere-768-l2|768|1000000"
    ["cohere-768-ip"]="https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings/documents-1m.hdf5.bz2|cohere-768-ip|768|1000000|--compressed"
    ["gist-960"]="http://ann-benchmarks.com/gist-960-euclidean.hdf5|gist|960|1000000"
    ["open-ai-1536"]="https://huggingface.co/datasets/navneet1v/datasets/resolve/main/open-ai-1536-temp.hdf5?download=true|open-ai-1536-temp|1536|1000000"
    ["bigAnn-10M-128"]="https://huggingface.co/datasets/navneet1v/datasets/resolve/main/bigann-10M-with-gt.hdf5?download=true|bigAnn-10M|128|10000000"
)

mkdir -p "$LOCAL_DIR"

for key in "${!DATASETS[@]}"; do
    IFS='|' read -r url name dim doc_count extra <<< "${DATASETS[$key]}"
    folder="${dim}_${doc_count}"
    vec_key="${BASE_PATH}/${folder}/${name}.knnvec"
    did_key="${BASE_PATH}/${folder}/${name}_ids.knndid"

    echo "=== Uploading ${name} (${dim}d, ${doc_count} docs) ==="
    python3 "$SCRIPT" \
        --download-url "$url" \
        --dataset-name "$name" \
        --bucket "$BUCKET" \
        --vectors-key "$vec_key" \
        --docids-key "$did_key" \
        --doc-count -1 \
        --region "$REGION" \
        --local-dir "$LOCAL_DIR" \
        $extra \
        $QUANTIZE_FLAG

    echo "Done: ${name}"
    echo ""
done

echo "All datasets uploaded."
