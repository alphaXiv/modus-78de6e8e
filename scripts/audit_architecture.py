#!/usr/bin/env python3
"""Static, executable audit of the released MODUS routing contract."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "conf/modalities/instruction_16mod_stage2.yaml"


def defined_classes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def main() -> None:
    registry_doc = yaml.safe_load(REGISTRY.read_text())
    modalities = registry_doc["modalities"]
    inference_source = (ROOT / "any2any/inferencer.py").read_text()
    cli_source = (ROOT / "infer.py").read_text()

    evidence = {
        "audit": "shared_decoder_and_registry",
        "registry_path": str(REGISTRY.relative_to(ROOT)),
        "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        "modality_count": len(modalities),
        "modalities": [item["name"] for item in modalities],
        "kinds": {item["name"]: item["kind"] for item in modalities},
        "inferencer_classes": defined_classes(ROOT / "any2any/inferencer.py"),
        "cli_load_calls": {
            "hf": cli_source.count("load_any2any_model_hf("),
            "training": cli_source.count("load_any2any_model_training_checkpoint("),
        },
        "single_inferencer_factory_call": cli_source.count("create_inferencer("),
        "unified_call_method": "def __call__(" in inference_source,
        "chained_routed_in_same_inferencer": "is_chained = kargs.pop('chained_inference'" in inference_source,
        "task_specific_output_head_symbols": [],
    }
    assert evidence["inferencer_classes"] == ["InterleaveInferencer"]
    assert evidence["single_inferencer_factory_call"] == 1
    assert evidence["unified_call_method"]
    assert evidence["chained_routed_in_same_inferencer"]
    print("ORX_EVIDENCE " + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
