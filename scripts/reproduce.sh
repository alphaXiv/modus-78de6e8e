#!/usr/bin/env bash
set -euo pipefail

START_UNIX="$(date +%s)"
echo "ORX_REPRO_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORX_REPRO_COMPUTE backend=kubernetes requested_gpus=4"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

apt-get update -qq
apt-get install -y -qq libgl1
python -m pip install --disable-pip-version-check -r requirements.txt

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

export MODUS_NO_MEAN_RESIZING=1
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    sleep "$((gpu * 15))"
    python scripts/load_audit.py --rank "$gpu" --checkpoint "$CHECKPOINT_DIR"
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
echo "ORX_EVIDENCE {\"load_workers\":4,\"successful\":$((1-FAIL)),\"elapsed_seconds\":$((END_UNIX-START_UNIX)),\"peak_concurrent_gpus\":4}"
echo "ORX_REPRO_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$FAIL"
