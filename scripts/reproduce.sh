#!/usr/bin/env bash
set -euo pipefail

START_UNIX="$(date +%s)"
echo "ORX_REPRO_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ORX_REPRO_COMPUTE backend=kubernetes gpu_count=0 reason=static_ast_yaml_audit"
python -m pip install --disable-pip-version-check "pyyaml==6.0.2"
python scripts/audit_architecture.py
END_UNIX="$(date +%s)"
echo "ORX_EVIDENCE {\"successful_static_audit\":true,\"elapsed_seconds\":$((END_UNIX-START_UNIX)),\"backend\":\"kubernetes\",\"gpu_count\":0}"
echo "ORX_REPRO_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
