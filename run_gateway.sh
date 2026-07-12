#!/usr/bin/env bash
set -euo pipefail

cd /vscratch/grp-rutaoyao/Harsh/multi_gpu_inference

source /user/htripath/venv/bin/activate

unset PYTHONPATH
unset LD_PRELOAD
unset LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1
export HF_HOME="/vscratch/grp-rutaoyao/Harsh/huggingface_cache"
export LD_LIBRARY_PATH="/opt/software/nvidia/lib64"

export NUM_GPUS="${NUM_GPUS:-2}"
export MODEL_ID="${MODEL_ID:-stabilityai/sd-turbo}"
export SCHED_POLICY="${SCHED_POLICY:-fifo}"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python - <<'PY'
import ctypes, torch
ctypes.CDLL("libcuda.so.1")
ctypes.CDLL("libnvidia-ml.so.1")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.device_count() >= 1, "No CUDA devices visible"
PY

exec uvicorn gateway:app --host 0.0.0.0 --port 8000
