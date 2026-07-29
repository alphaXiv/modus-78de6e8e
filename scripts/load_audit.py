#!/usr/bin/env python3
"""Load one complete released runtime per GPU and emit auditable evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from any2any.any2any_tasks import create_inferencer
from any2any.load_any2any import load_any2any_model_hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, vae, tokenizer, token_ids, registry = load_any2any_model_hf(
        model_path=args.checkpoint,
        modality_config_path="conf/modalities/instruction_16mod_stage2.yaml",
    )
    inferencer = create_inferencer(
        model=model,
        vae_model=vae,
        tokenizer=tokenizer,
        new_token_ids=token_ids,
        modality_registry=registry,
    )
    torch.cuda.synchronize()
    marker = Path(f"/tmp/modus_loaded_{args.rank}")
    marker.write_text("loaded")

    barrier_started = time.monotonic()
    while len(list(Path("/tmp").glob("modus_loaded_*"))) < 4:
        if time.monotonic() - barrier_started > 300:
            raise TimeoutError("four-GPU load barrier timed out")
        time.sleep(1)

    payload = {
        "claim": "released_environment_runtime_load",
        "rank": args.rank,
        "checkpoint": "EPFL-VILAB/MODUS",
        "model_class": type(model).__name__,
        "vae_class": type(vae).__name__,
        "inferencer_class": type(inferencer).__name__,
        "registry_modalities": [spec.name for spec in registry.specs],
        "modality_count": len(registry.specs),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
        "all_four_loaded_concurrently": True,
    }
    print("ORX_EVIDENCE " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
