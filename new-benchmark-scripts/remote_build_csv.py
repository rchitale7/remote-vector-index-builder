#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import csv
import re
import subprocess
import time
import threading
import os
from datetime import datetime
import boto3
import requests
import json
import psutil
import pandas as pd
from py3nvml import nvidia_smi

DOCKER_IMAGE = "opensearchstaging/remote-vector-index-builder:api-snapshot"
CONTAINER_NAME = "remote-index-builder"

TIMING_PATTERNS = [
    ("vector_download_time", r"Vector download time for vector path .+?: (.+)"),
    ("index_build_time", r"Index build time for vector path .+?: (.+)"),
    ("index_conversion_time", r"Index conversion time for vector path .+?: (.+)"),
    ("index_write_time", r"Index write time for vector path .+?: (.+)"),
    ("total_index_build_time", r"Total index build time for path .+?: (.+)"),
    ("upload_time", r"Total upload time for path .+?: (.+)"),
]
LOG_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
ADDED_JOB_PATTERN = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Added job id"
COMPLETED_JOB_PATTERN = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*completed with status: (\S+),"


class MemoryMonitor:
    _nvml_initialized = False
    _nvml_handle = None

    def __init__(self, identifier, interval=0.1):
        self.interval = interval
        self.identifier = identifier
        self.cpu_memory_logs = []
        self.gpu_memory_logs = []
        self.is_monitoring = False
        self._monitor_thread = None
        self.monitor_gpu = True

        if not MemoryMonitor._nvml_initialized:
            try:
                nvidia_smi.nvmlInit()
                MemoryMonitor._nvml_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                MemoryMonitor._nvml_initialized = True
            except Exception:
                self.monitor_gpu = False

        self.handle = MemoryMonitor._nvml_handle
        if not self.handle:
            self.monitor_gpu = False

    def _monitoring_loop(self):
        while self.is_monitoring:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cpu_memory_logs.append({
                "cpu_used_system_memory": psutil.virtual_memory().used / 1024 / 1024,
                "cur_time": cur_time,
            })
            if self.monitor_gpu:
                try:
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                    self.gpu_memory_logs.append({
                        "gpu_used_system_memory": info.used / 1024 / 1024,
                        "cur_time": cur_time,
                    })
                except Exception:
                    pass
            time.sleep(self.interval)

    def start(self):
        self.cpu_memory_logs = []
        self.gpu_memory_logs = []
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def report(self):
        results = {}
        if self.cpu_memory_logs:
            df = pd.DataFrame(self.cpu_memory_logs)
            peak = df["cpu_used_system_memory"].max()
            net = peak - df["cpu_used_system_memory"].iloc[0]
            results["cpu_peak_mb"] = round(peak, 2)
            results["cpu_net_mb"] = round(net, 2)

        if self.monitor_gpu and self.gpu_memory_logs:
            df = pd.DataFrame(self.gpu_memory_logs)
            peak = df["gpu_used_system_memory"].max()
            net = peak - df["gpu_used_system_memory"].iloc[0]
            results["gpu_peak_mb"] = round(peak, 2)
            results["gpu_net_mb"] = round(net, 2)

        return results


def get_docker_logs():
    result = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
    return (result.stdout + result.stderr).splitlines()


def parse_logs_for_job(vec_path, all_logs):
    relevant = [line for line in all_logs if vec_path in line]
    metrics = {}

    for key, pattern in TIMING_PATTERNS:
        for line in relevant:
            m = re.search(pattern, line)
            if m:
                metrics[key] = m.group(1).strip()
                break

    start_time = end_time = None
    for line in relevant:
        m = re.search(ADDED_JOB_PATTERN, line)
        if m and not start_time:
            start_time = datetime.strptime(m.group(1), LOG_TIMESTAMP_FMT)
        m = re.search(COMPLETED_JOB_PATTERN, line)
        if m:
            end_time = datetime.strptime(m.group(1), LOG_TIMESTAMP_FMT)
            metrics["status"] = m.group(2)

    if start_time and end_time:
        metrics["total_time"] = (end_time - start_time).total_seconds()

    return metrics


def wait_for_job(vec_path, timeout=3600, poll_interval=5):
    start = time.time()
    while time.time() - start < timeout:
        logs = get_docker_logs()
        for line in logs:
            if vec_path in line and "completed with status:" in line:
                return "COMPLETED"
            if vec_path in line and ("FAILED" in line or "error" in line.lower()):
                return "FAILED"
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True
        )
        if result.stdout.strip() != "true":
            return "CONTAINER_EXITED"
        time.sleep(poll_interval)
    return "TIMEOUT"


def main():
    parser = argparse.ArgumentParser(description="Remote vector index build from CSV")
    parser.add_argument("-b", "--s3-bucket", required=True)
    parser.add_argument("-s", "--s3-base-path", required=True, help="S3 base path (e.g. datasets/float)")
    parser.add_argument("-t", "--data-type", required=True, choices=["float", "half_float", "binary"])
    parser.add_argument("-c", "--csv", required=True, help="CSV file listing datasets")
    parser.add_argument("-g", "--graph-only", action="store_true")
    parser.add_argument("-o", "--output", default="results.csv", help="Output CSV file")
    args = parser.parse_args()

    with open(args.csv) as f:
        datasets = list(csv.DictReader(f))

    # Stop any existing container
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    print("Starting docker container...")
    proc = subprocess.Popen([
        "docker", "run", "--name", CONTAINER_NAME,
        "--env-file", ".dockerenv",
        "--gpus", "all",
        "-p", "80:1025",
        DOCKER_IMAGE
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(30):
        try:
            requests.get("http://0.0.0.0:80/_heart_beat")
            print("Container is ready.")
            break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("Container failed to start. Dumping logs:")
        subprocess.run(["docker", "logs", CONTAINER_NAME])
        proc.terminate()
        exit(1)

    s3_base = args.s3_base_path.strip("/")
    all_results = []

    for ds in datasets:
        basename = os.path.splitext(ds['filename'])[0]
        vec_path = f"{s3_base}/{basename}.knnvec"
        did_path = f"{s3_base}/{basename}_ids.knndid"
        dimension = int(ds['dimensions'])
        doc_count = int(ds['doc_count'])
        space_type = ds['space_type']

        payload = {
            "repository_type": "s3",
            "container_name": args.s3_bucket,
            "vector_path": vec_path,
            "doc_id_path": did_path,
            "dimension": dimension,
            "doc_count": doc_count,
            "data_type": args.data_type,
            "index_parameters": {
                "space_type": space_type,
            },
            "graph_only": args.graph_only,
        }

        monitor = MemoryMonitor(identifier=basename)
        monitor.start()

        print(f"\nBuilding index for {basename} ({dimension}d, {doc_count} docs, {space_type}) ...")
        r = requests.post("http://0.0.0.0:80/_build", json=payload)
        resp_json = r.json()
        print(json.dumps(resp_json, indent=2))

        job_id = resp_json.get("job_id")
        if job_id:
            print(f"Waiting for job {job_id} ...")
            status = wait_for_job(vec_path)
            print(f"Job status: {status}")

        monitor.stop()
        mem = monitor.report()

        logs = get_docker_logs()
        log_metrics = parse_logs_for_job(vec_path, logs)

        row = {
            "dataset": basename,
            "dimension": dimension,
            "doc_count": doc_count,
            "data_type": args.data_type,
            "space_type": space_type,
            "graph_only": args.graph_only,
        }
        row.update(log_metrics)
        row.update(mem)
        all_results.append(row)

        print(f"  {row}")
        print(f"Done processing {basename}\n")

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nResults written to {args.output}")

    print("\n=== Full Docker Container Logs ===")
    subprocess.run(["docker", "logs", CONTAINER_NAME])

    proc.terminate()


if __name__ == "__main__":
    main()
