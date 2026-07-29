#!/usr/bin/env bash
set -euo pipefail

START_UNIX="$(date +%s)"
echo "ORX_REPRO_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORX_REPRO_COMPUTE backend=kubernetes requested_gpus=4"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

python -m pip install --disable-pip-version-check \
  "huggingface_hub==0.29.1" "omegaconf>=2.3" "hydra-core>=1.3" \
  "transformers==4.49.0" "safetensors==0.4.5" "accelerate>=0.34" \
  "einops==0.8.1" "numpy==1.24.4" "pillow>=9.3" \
  "opencv-python-headless==4.7.0.72" "scipy==1.10.1" \
  "scikit-learn==1.2.2" "matplotlib==3.7.0" "pyyaml>=6"

python scripts/audit_architecture.py

CHECKPOINT_DIR=/workspace/cache/modus
mkdir -p "$CHECKPOINT_DIR"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "EPFL-VILAB/MODUS",
    local_dir="/workspace/cache/modus",
    local_dir_use_symlinks=False,
)
print(f"ORX_CHECKPOINT repo=EPFL-VILAB/MODUS path={path}")
PY

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/workspace/cache/modus")
files = []
for path in sorted(root.iterdir()):
    if path.is_file():
        files.append({"name": path.name, "bytes": path.stat().st_size})
config_sha = hashlib.sha256((root / "llm_config.json").read_bytes()).hexdigest()
print("ORX_EVIDENCE " + json.dumps({
    "checkpoint": "EPFL-VILAB/MODUS",
    "files": files,
    "llm_config_sha256": config_sha,
}, sort_keys=True))
PY

mkdir -p /workspace/outputs

run_task() {
  local gpu="$1"
  local label="$2"
  shift 2
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "ORX_TASK_START label=$label gpu=$gpu utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python infer.py "$@" \
      checkpoint_path="$CHECKPOINT_DIR" \
      format=hf image_size=512 num_timesteps=5 num_samples=1 seed=20260729 \
      output_dir="/workspace/outputs/$label"
    echo "ORX_TASK_END label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) 2>&1 | sed -u "s/^/[$label] /"
}

run_task 0 rgb_depth --condition rgb --target depth input_image=test_images/01_basil_cathedral.jpg &
PID0=$!
run_task 1 rgb_normal --condition rgb --target normal input_image=test_images/02_waterfall.jpg &
PID1=$!
run_task 2 rgb_canny_normal --condition rgb --target normal --intermediate canny input_image=test_images/03_kremlin_clock.jpg &
PID2=$!
run_task 3 rgb_caption --condition rgb --target caption input_image=test_images/06_city_bus.jpg &
PID3=$!

FAIL=0
for pid in "$PID0" "$PID1" "$PID2" "$PID3"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

END_UNIX="$(date +%s)"
echo "ORX_EVIDENCE {\"scout_tasks\":4,\"successful\":$((1-FAIL)),\"elapsed_seconds\":$((END_UNIX-START_UNIX)),\"peak_concurrent_gpus\":4}"
echo "ORX_REPRO_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$FAIL"
