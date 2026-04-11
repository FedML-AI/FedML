#!/usr/bin/env bash
# gpu_wrapper.sh — Per-process CUDA isolation for multi-GPU MPI experiments.
#
# Problem: With CUDA_VISIBLE_DEVICES=1,2,3, every MPI process initializes
# CUDA contexts on ALL visible GPUs (~428 MiB each), causing cross-GPU
# memory pollution that triggers OOM on shared GPUs.
#
# Solution: Before the Python process starts, restrict CUDA_VISIBLE_DEVICES
# to the single physical GPU assigned to this MPI rank.
#
# Required env vars (set by run_experiment.sh):
#   GPU_PROC_MAPPING  — comma-separated process counts per GPU, e.g. "13,19,19"
#   GPU_PHYS_IDS      — comma-separated physical GPU IDs,      e.g. "1,2,3"
#
# Usage: mpirun -np 51 bash scripts/gpu_wrapper.sh main_fedml_shieldfl.py --cf ...

set -euo pipefail

RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"

IFS=',' read -ra MAPPING <<< "${GPU_PROC_MAPPING}"
IFS=',' read -ra GPUS    <<< "${GPU_PHYS_IDS}"

offset=0
assigned_gpu=""
for i in "${!MAPPING[@]}"; do
    next=$((offset + MAPPING[i]))
    if [ "$RANK" -lt "$next" ]; then
        assigned_gpu="${GPUS[$i]}"
        break
    fi
    offset=$next
done

if [ -z "$assigned_gpu" ]; then
    echo "ERROR: gpu_wrapper.sh — rank $RANK exceeds mapping sum" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$assigned_gpu"
exec python "$@"
