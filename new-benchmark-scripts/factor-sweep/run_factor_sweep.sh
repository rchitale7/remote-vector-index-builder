#!/usr/bin/env bash
#
# run_factor_sweep.sh — CPU get_cpus() factor sweep wrapper.
#
# For each pre-built factor image, re-point the ':api-snapshot' tag at it (so the
# unmodified remote_build_csv.py picks it up) and run the build benchmark REPS
# times. Each rep writes results/<label>/run_NN.csv. When all runs finish it
# invokes aggregate_factors.py to produce p50/p90/range for download & upload.
#
# Prereqs (same as remote_build_csv.py):
#   - the six api-snapshot-<label> images built (baseline, f1, f1_5, f2, f2_5, f3)
#   - .dockerenv present in the benchmark-scripts dir (AWS creds + LOG_LEVEL=DEBUG)
#   - datasets staged to S3, GPU host with docker + nvidia runtime
#
# Usage:
#   ./run_factor_sweep.sh -b <bucket> -s datasets/float -t float [-c datasets.csv] [-r 10]
#
set -euo pipefail

REPO_IMAGE="opensearchstaging/remote-vector-index-builder"
LIVE_TAG="${REPO_IMAGE}:api-snapshot"      # tag remote_build_csv.py always launches

# label -> image tag suffix. "baseline" == mainline 0.625/0.5; others symmetric.
LABELS=(baseline f1 f1_5 f2 f2_5 f3)

# ---- args -------------------------------------------------------------------
BUCKET="" ; S3_PATH="" ; DTYPE="" ; CSV="datasets.csv" ; REPS=10 ; OUTDIR="results"
usage() {
  echo "Usage: $0 -b <bucket> -s <s3-base-path> -t <float|half_float|binary>" \
       "[-c datasets.csv] [-r reps] [-o results_dir]" >&2
  exit 1
}
while getopts "b:s:t:c:r:o:h" opt; do
  case "$opt" in
    b) BUCKET="$OPTARG" ;;
    s) S3_PATH="$OPTARG" ;;
    t) DTYPE="$OPTARG" ;;
    c) CSV="$OPTARG" ;;
    r) REPS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h|*) usage ;;
  esac
done
[[ -z "$BUCKET" || -z "$S3_PATH" || -z "$DTYPE" ]] && usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"        # new-benchmark-scripts/
BUILD_CSV="${BENCH_DIR}/remote_build_csv.py"

# Python interpreter with the benchmark deps (boto3/requests/pandas/psutil/
# py3nvml/numpy). Override with PYTHON=... ; defaults to the my_env conda env if
# present, else falls back to whatever `python` is on PATH.
DEFAULT_PY="$HOME/miniconda3/envs/my_env/bin/python"
PYTHON="${PYTHON:-$([[ -x "$DEFAULT_PY" ]] && echo "$DEFAULT_PY" || echo python)}"

[[ -f "$BUILD_CSV" ]] || { echo "ERROR: not found: $BUILD_CSV" >&2; exit 1; }
[[ -f "${BENCH_DIR}/.dockerenv" ]] || { echo "ERROR: ${BENCH_DIR}/.dockerenv missing" >&2; exit 1; }
[[ -f "${BENCH_DIR}/${CSV}" ]] || { echo "ERROR: ${BENCH_DIR}/${CSV} missing" >&2; exit 1; }

# Fail fast if the chosen interpreter is missing required modules, rather than
# crashing partway through a multi-hour sweep.
if ! "$PYTHON" -c "import boto3, requests, pandas, numpy, psutil, py3nvml" 2>/dev/null; then
  echo "ERROR: '$PYTHON' is missing required modules." >&2
  echo "       Activate the benchmark conda env or set PYTHON=/path/to/env/bin/python" >&2
  "$PYTHON" -c "import boto3, requests, pandas, numpy, psutil, py3nvml" || true
  exit 1
fi
echo "Using interpreter: $PYTHON"

# Verify all images exist up front so we fail fast rather than mid-sweep.
for label in "${LABELS[@]}"; do
  img="${REPO_IMAGE}:api-snapshot-${label}"
  docker image inspect "$img" >/dev/null 2>&1 || { echo "ERROR: image missing: $img" >&2; exit 1; }
done

mkdir -p "${SCRIPT_DIR}/${OUTDIR}"
echo "Sweep: labels=[${LABELS[*]}] reps=${REPS} csv=${CSV} bucket=${BUCKET} path=${S3_PATH} type=${DTYPE}"

for label in "${LABELS[@]}"; do
  img="${REPO_IMAGE}:api-snapshot-${label}"
  echo ""
  echo "############################################################"
  echo "# ${label}  ->  ${img}"
  echo "############################################################"
  # Point the tag remote_build_csv.py launches at this factor's image.
  docker tag "$img" "$LIVE_TAG"

  label_dir="${SCRIPT_DIR}/${OUTDIR}/${label}"
  mkdir -p "$label_dir"

  for rep in $(seq 1 "$REPS"); do
    out="${label_dir}/run_$(printf '%02d' "$rep").csv"
    echo ""
    echo "==== ${label} rep ${rep}/${REPS} -> ${out} ===="
    # Run from the benchmark dir so .dockerenv / relative paths resolve.
    ( cd "$BENCH_DIR" && "$PYTHON" remote_build_csv.py \
        -b "$BUCKET" -s "$S3_PATH" -t "$DTYPE" -c "$CSV" -o "$out" ) \
      || echo "WARN: ${label} rep ${rep} exited non-zero; continuing"
  done
done

echo ""
echo "############################################################"
echo "# Aggregating"
echo "############################################################"
"$PYTHON" "${SCRIPT_DIR}/aggregate_factors.py" \
  --results-dir "${SCRIPT_DIR}/${OUTDIR}" \
  --output "${SCRIPT_DIR}/${OUTDIR}/factor_summary.csv"

echo ""
echo "Sweep complete. Per-rep CSVs under ${SCRIPT_DIR}/${OUTDIR}/<label>/, summary at ${SCRIPT_DIR}/${OUTDIR}/factor_summary.csv"
