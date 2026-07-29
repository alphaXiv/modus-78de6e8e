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
  "scikit-learn==1.2.2" "matplotlib==3.7.0" "pyyaml>=6" \
  "pyarrow==11.0.0"

python scripts/audit_architecture.py

CHECKPOINT_DIR=/workspace/cache/modus
DATASET_DIR=/workspace/cache/nyuv2
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

mkdir -p "$DATASET_DIR"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "tanganke/nyuv2",
    repo_type="dataset",
    allow_patterns=["data/val-*.parquet", "README.md"],
    local_dir="/workspace/cache/nyuv2",
)
print(f"ORX_DATASET repo=tanganke/nyuv2 split=val expected_examples=654 path={path}")
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

export MODUS_NO_MEAN_RESIZING=1
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python scripts/nyuv2_eval.py \
      --rank "$gpu" --world-size 4 \
      --checkpoint "$CHECKPOINT_DIR" \
      --dataset-dir "$DATASET_DIR" \
      --steps 5
  ) >"/tmp/rank_${gpu}.log" 2>&1 &
  eval "PID${gpu}=$!"
done

FAIL=0
for pid in "$PID0" "$PID1" "$PID2" "$PID3"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done

for gpu in 0 1 2 3; do
  echo "ORX_RANK_LOG_BEGIN rank=$gpu"
  cat "/tmp/rank_${gpu}.log"
  echo "ORX_RANK_LOG_END rank=$gpu"
done

END_UNIX="$(date +%s)"
echo "ORX_EVIDENCE {\"scout_tasks\":4,\"successful\":$((1-FAIL)),\"elapsed_seconds\":$((END_UNIX-START_UNIX)),\"peak_concurrent_gpus\":4}"
echo "ORX_REPRO_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$FAIL"
