# Benchmark Scripts

CSV-driven benchmarking harness for the Remote Vector Index Builder. These scripts
let you stage datasets to S3, build indices end-to-end through the API docker
container (capturing per-stage timing and memory), and measure search recall and
throughput — all driven by a single dataset manifest (`datasets.csv`).

## Primary purpose: benchmarking local code changes

The main reason these scripts exist is to measure the performance impact of **your
local changes** to the index builder. The `remote_build_csv.py` benchmark runs against
a docker image, so the workflow is: change code → rebuild the image → run the
benchmark.

`remote_build_csv.py` starts the image tagged
`opensearchstaging/remote-vector-index-builder:api-snapshot` (set via `DOCKER_IMAGE` at
the top of the script). To test local changes you must build **your modified code**
into that tag before running the benchmark.

The API image is layered on top of the `core` image, which is layered on the
`faiss-base` image (see the [DEVELOPER_GUIDE](../DEVELOPER_GUIDE.md#developer-guide)).
Rebuild only the layer(s) you changed, then tag the final API image as `api-snapshot`.
Run all `docker build` commands from the **repository root**.

```bash
cd ..   # repository root: remote-vector-index-builder/

# If you changed code under remote_vector_index_builder/app/ (the API layer):
docker build -f ./remote_vector_index_builder/app/Dockerfile . \
  -t opensearchstaging/remote-vector-index-builder:api-snapshot

# If you changed code under remote_vector_index_builder/core/ (the core build logic),
# rebuild core first, then rebuild the API image on top of it:
docker build -f ./remote_vector_index_builder/core/Dockerfile . \
  -t opensearchstaging/remote-vector-index-builder:core-snapshot
docker build -f ./remote_vector_index_builder/app/Dockerfile . \
  --build-arg CORE_IMAGE_TAG=core-snapshot \
  -t opensearchstaging/remote-vector-index-builder:api-snapshot
```

> The core changes most relevant to build performance (index build, GPU→CPU
> conversion, download/upload) live under `remote_vector_index_builder/core/` — e.g.
> `core/tasks.py` and `core/index_builder/faiss/`. Changing those requires rebuilding
> the `core` image and then the `api` image on top of it, as shown above.

> The `faiss-base` image rarely changes and takes a long time to build. Only rebuild it
> if you modified the Faiss submodule; otherwise reuse the published/base image (see the
> DEVELOPER_GUIDE).

Once the `api-snapshot` image contains your changes, run `remote_build_csv.py`
(section 2) to benchmark them. To compare against the baseline, build the unmodified
code under a different tag, run the benchmark, and diff the two output CSVs.

## Contents

| Script | Purpose |
| --- | --- |
| `dataset_to_local.py` | Convert local HDF5 datasets to `.knnvec` / `_ids.knndid` and upload to S3 (float or one-bit quantized binary). |
| `remote_build_csv.py` | Run the API docker container and build an index per dataset, capturing download / build / conversion / write / upload times, total time, and peak CPU/GPU memory into a CSV. |
| `recall_test_csv.py` | Download built indices from S3 and measure recall@k, recall@1, recall@100, search time, and throughput per dataset. |
| `datasets.csv` | Dataset manifest consumed by all three scripts. |

## The dataset manifest: `datasets.csv`

Every script reads this file. The current manifest ships with seven datasets:

```csv
filename,normalize,doc_count,dimensions,space_type
sift.hdf5,false,1000000,128,l2
ms-marco-384.hdf5,false,1000000,384,l2
cohere-768-l2.hdf5,false,1000000,768,l2
cohere-768-ip.hdf5,false,1000000,768,innerproduct
gist.hdf5,false,1000000,960,l2
open-ai-1536-temp.hdf5,false,1000000,1536,l2
bigAnn-10M.hdf5,false,10000000,128,l2
```

- `filename` — HDF5 file name (basename is reused for the S3 keys and index name).
- `normalize` — `true`/`false`, whether to L2-normalize vectors on load.
- `doc_count` — number of vectors.
- `dimensions` — vector dimensionality.
- `space_type` — `l2` or `innerproduct`.

To benchmark **a single dataset**, make a CSV with just the header and one row.
For example, create `one.csv`:

```csv
filename,normalize,doc_count,dimensions,space_type
sift.hdf5,false,1000000,128,l2
```

Then pass `--csv one.csv` (or `-c one.csv`) to any of the scripts below.

## Prerequisites

- Python env with: `boto3`, `requests`, `psutil`, `pandas`, `numpy`, `faiss`, `tqdm`,
  `py3nvml`, `h5py`, plus the repo's `benchmarking` package importable (the scripts add
  it to `sys.path`). The conda environment described in the top-level
  [`scripts/README.md`](../scripts/README.md) works.
- AWS credentials configured for the target S3 bucket.
- A GPU host with Docker and the NVIDIA container runtime (for `remote_build_csv.py`).
- Local HDF5 dataset files (for `dataset_to_local.py` and `recall_test_csv.py`).

---

## 0. Download the datasets

`dataset_to_local.py` and `recall_test_csv.py` read HDF5 files from a **local
directory** — they do not download anything. You must fetch the HDF5 files first and
place them in your `--input-dir` / `--datasets-dir`, saving each one under the exact
`filename` listed in `datasets.csv`.

Every dataset in the shipped `datasets.csv` has a download link in
[`../benchmarking/benchmarks.yml`](../benchmarking/benchmarks.yml). The table below maps
each `datasets.csv` row to its source and the filename you must save it as:

| `datasets.csv` filename | Download URL | Save as / notes |
| --- | --- | --- |
| `sift.hdf5` | `https://ann-benchmarks.com/sift-128-euclidean.hdf5` | Rename download to `sift.hdf5`. |
| `ms-marco-384.hdf5` | `https://huggingface.co/datasets/navneet1v/datasets/resolve/main/ms_marco-384-1m.hdf5?download=true` | Downloads as `ms_marco-384-1m.hdf5`; rename to `ms-marco-384.hdf5`. |
| `cohere-768-l2.hdf5` | `https://huggingface.co/datasets/navneet1v/datasets/resolve/main/cohere-768-l2.hdf5?download=true` | Name matches directly. |
| `cohere-768-ip.hdf5` | `https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings/documents-1m.hdf5.bz2` | `bz2` — decompress, then rename to `cohere-768-ip.hdf5`. |
| `gist.hdf5` | `http://ann-benchmarks.com/gist-960-euclidean.hdf5` | Rename download to `gist.hdf5`. |
| `open-ai-1536-temp.hdf5` | `https://huggingface.co/datasets/navneet1v/datasets/resolve/main/open-ai-1536-temp.hdf5?download=true` | Name matches directly. |
| `bigAnn-10M.hdf5` | `https://huggingface.co/datasets/navneet1v/datasets/resolve/main/bigann-10M-with-gt.hdf5?download=true` | Downloads as `bigann-10M-with-gt.hdf5`; rename to `bigAnn-10M.hdf5`. |

Example — download `sift` and `gist` and save them under the CSV filenames:

```bash
mkdir -p ../../datasets
cd ../../datasets
wget -O sift.hdf5 https://ann-benchmarks.com/sift-128-euclidean.hdf5
wget -O gist.hdf5 http://ann-benchmarks.com/gist-960-euclidean.hdf5
cd -
```

For the `bz2` dataset (`cohere-768-ip`), download, decompress, then rename:

```bash
cd ../../datasets
wget https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings/documents-1m.hdf5.bz2
bunzip2 documents-1m.hdf5.bz2
mv documents-1m.hdf5 cohere-768-ip.hdf5
cd -
```

> **Filename must match the CSV.** The scripts locate each file by the `filename`
> column in `datasets.csv`. Several downloads arrive under a different name (e.g.
> `ms_marco-384-1m.hdf5`, `bigann-10M-with-gt.hdf5`), so rename them to match the CSV
> row (or edit the CSV so they agree). Trim `datasets.csv` to just the rows you have
> HDF5 files for.

Each HDF5 file is expected to contain a `train` dataset (indexing vectors), a `test`
dataset (query vectors), and a `neighbors` dataset (ground truth) — the standard
ann-benchmarks layout used by `benchmarking/dataset/dataset_utils.py`.

---

## 1. Stage datasets to S3 — `dataset_to_local.py`

Reads the CSV, loads each local HDF5, converts vectors to binary `.knnvec` and doc
IDs to `_ids.knndid`, and uploads to `s3://<bucket>/<s3-prefix>/`.

Float32 vectors:

```bash
python dataset_to_local.py \
  --csv datasets.csv \
  --input-dir ../../datasets \
  --bucket <your-bucket> \
  --s3-prefix datasets/float \
  --region us-east-1
```

One-bit scalar-quantized (binary) vectors — matches k-NN `OneBitScalarQuantizer` + `BitPacker`:

```bash
python dataset_to_local.py \
  --csv datasets.csv \
  --input-dir ../../datasets \
  --bucket <your-bucket> \
  --s3-prefix datasets/binary \
  --quantize
```

Single dataset: point `--csv` at a one-row CSV as shown above.

| Flag | Description |
| --- | --- |
| `--csv` | Dataset manifest. |
| `--input-dir` | Directory containing the HDF5 files. |
| `--bucket` | Target S3 bucket. |
| `--s3-prefix` | Key prefix, e.g. `datasets/float` or `datasets/binary`. |
| `--region` | AWS region (default `us-east-1`). |
| `--quantize` | Apply one-bit scalar quantization (binary output). |

---

## 2. End-to-end build benchmark — `remote_build_csv.py`

Starts the API container, submits a `/_build` request per dataset, monitors memory,
waits for completion by tailing container logs, and parses per-stage timings into a
results CSV.

### Set up the container environment file

The script launches the container with `--env-file .dockerenv`. Create a `.dockerenv`
file in this directory. For real AWS S3:

```bash
cat > .dockerenv <<'EOF'
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_DEFAULT_REGION=us-east-1
LOG_LEVEL=DEBUG
EOF
```

> `LOG_LEVEL=DEBUG` is **required** — the per-stage timing columns are parsed from the
> container's DEBUG log lines. Without it, only `total_time`, `status`, and memory
> columns are populated.

> `.dockerenv` contains credentials. Do not commit it; add it to `.gitignore`.

### Run

Float, full index (graph + vectors):

```bash
python remote_build_csv.py \
  -b <your-bucket> \
  -s datasets/float \
  -t float \
  -c datasets.csv \
  -o build_results_float.csv
```

Binary vectors:

```bash
python remote_build_csv.py \
  -b <your-bucket> \
  -s datasets/binary \
  -t binary \
  -c datasets.csv \
  -o build_results_binary.csv
```

Graph-only build (skips storing vectors in the index — the "no stitch" case):

```bash
python remote_build_csv.py \
  -b <your-bucket> \
  -s datasets/float \
  -t float \
  -c datasets.csv \
  -g \
  -o build_results_skip_stored_vectors.csv
```

Single dataset: use a one-row `--csv`/`-c` file.

| Flag | Description |
| --- | --- |
| `-b`, `--s3-bucket` | S3 bucket holding the staged vectors. |
| `-s`, `--s3-base-path` | S3 base prefix, e.g. `datasets/float` (must match step 1). |
| `-t`, `--data-type` | `float`, `half_float`, or `binary`. |
| `-c`, `--csv` | Dataset manifest. |
| `-g`, `--skip-stored-vectors` | Build graph only; skip storing vectors in the index (sends `skip_stored_vectors=true` to `/_build`). |
| `-o`, `--output` | Output CSV (default `results.csv`). |

### Output columns

- Identity: `dataset`, `dimension`, `doc_count`, `data_type`, `space_type`, `skip_stored_vectors`
- Timings (seconds, from container DEBUG logs): `vector_download_time`,
  `index_build_time`, `index_conversion_time`, `index_write_time`,
  `total_index_build_time`, `upload_time`, and `total_time` (job add → completion)
- Status: `status`
- Memory (MB): `cpu_peak_mb`, `cpu_net_mb`, `gpu_peak_mb`, `gpu_net_mb`

### Notes

- The container is named `remote-index-builder` and published on host port `80`
  (`http://0.0.0.0:80`). Any existing container with that name is removed first.
- Uses image `opensearchstaging/remote-vector-index-builder:api-snapshot`. Build/pull it
  beforehand, or edit `DOCKER_IMAGE` at the top of the script.
- Requires `--gpus all`; run on a GPU host.
- Full container logs are printed at the end of the run.

---

## 3. Recall & throughput benchmark — `recall_test_csv.py`

Downloads each built `.faiss` graph (and vectors, when needed) from S3, reconstructs a
searchable index, runs the query set, and reports recall and throughput.

Supported graph/vector combinations:

| `--graph-type` | `--vector-type` | Behavior |
| --- | --- | --- |
| `float` | `float` | Float HNSW with float storage (no stitching). |
| `float` | `binary` | Copy the FP32 HNSW graph into `IndexBinaryHNSW` + BQ storage. |
| `binary` | `binary` | Load `IndexBinaryHNSW` directly. |

FP32 graph + FP32 vectors (the "no stitch" case):

```bash
python recall_test_csv.py \
  --csv datasets.csv \
  --bucket <your-bucket> \
  --graph-path datasets/float \
  --vector-path datasets/float \
  --graph-type float --vector-type float \
  --datasets-dir ../../datasets \
  --ef-search 256 --k 100 \
  -o recall_float.csv
```

FP32 graph + binary (BQ) vectors:

```bash
python recall_test_csv.py \
  --csv datasets.csv \
  --bucket <your-bucket> \
  --graph-path datasets/float \
  --vector-path datasets/binary \
  --graph-type float --vector-type binary \
  --datasets-dir ../../datasets \
  --ef-search 256 --k 100 \
  -o recall_fp32graph_bq.csv
```

Single dataset: use a one-row `--csv` file.

| Flag | Description |
| --- | --- |
| `--csv` | Dataset manifest. |
| `--bucket` | S3 bucket. |
| `--graph-path` | S3 prefix for `.faiss` graphs, e.g. `datasets/float`. |
| `--vector-path` | S3 prefix for `.knnvec` vectors, e.g. `datasets/float` or `datasets/binary`. |
| `--graph-type` | `float` or `binary`. |
| `--vector-type` | `float` or `binary`. |
| `--datasets-dir` | Local dir with HDF5 files (for queries + ground truth). |
| `--m` | HNSW M (default `16`), used when stitching a binary index. |
| `--ef-search` | efSearch (default `256`). |
| `--k` | k for recall (default `100`). |
| `--region` | AWS region (default `us-east-1`). |
| `-o`, `--output` | Output CSV (default `recall_results.csv`). |

### Output columns

`dataset`, `dimension`, `doc_count`, `space_type`, `graph_type`, `vector_type`,
`recall@<k>`, `recall@1`, `recall@100`, `search_time_s`, `throughput_qps`,
`num_queries`, `ef_search`.

---

## Typical end-to-end workflow

```bash
# 1. Stage float vectors to S3
python dataset_to_local.py --csv datasets.csv --input-dir ../../datasets \
  --bucket my-bucket --s3-prefix datasets/float

# (optional) stage binary-quantized vectors too
python dataset_to_local.py --csv datasets.csv --input-dir ../../datasets \
  --bucket my-bucket --s3-prefix datasets/binary --quantize

# 2. Build indices via the docker container, capturing timing + memory
python remote_build_csv.py -b my-bucket -s datasets/float -t float \
  -c datasets.csv -o build_results.csv

# 3. Measure recall & throughput of the built indices
python recall_test_csv.py --csv datasets.csv --bucket my-bucket \
  --graph-path datasets/float --vector-path datasets/float \
  --graph-type float --vector-type float \
  --datasets-dir ../../datasets -o recall_results.csv
```
