#!/usr/bin/env python3
"""Paired released-checkpoint evaluation on a preregistered NYUv2 slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from any2any.any2any_tasks import (
    _build_understanding_hyper,
    create_inferencer,
    generate_any,
)
from any2any.load_any2any import load_any2any_model_hf


SLICE_INDICES = [0, 41, 83, 125, 167, 209, 251, 293, 335, 377, 419, 461, 503, 545, 587, 629]
SEED_BASE = 20260729


def emit(payload: dict) -> None:
    print("ORX_EVIDENCE " + json.dumps(payload, sort_keys=True), flush=True)


class Nyuv2Validation:
    def __init__(self, dataset_dir: str):
        self.files = sorted(Path(dataset_dir).glob("data/val-*.parquet"))
        self.offsets = []
        total = 0
        for path in self.files:
            rows = pq.ParquetFile(path).metadata.num_rows
            self.offsets.append((total, total + rows, path))
            total += rows
        self.length = total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        for start, end, path in self.offsets:
            if start <= index < end:
                local_index = index - start
                table = pq.read_table(path, columns=["image", "normal"])
                return {
                    "image": table["image"][local_index].as_py(),
                    "normal": table["normal"][local_index].as_py(),
                }
        raise IndexError(index)


def as_rgb(sample_image: np.ndarray) -> Image.Image:
    array = np.asarray(sample_image)
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f":
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def angular_mae(prediction: Image.Image, target: np.ndarray) -> tuple[float, int]:
    gt = np.asarray(target, dtype=np.float32)
    if gt.shape[0] == 3:
        gt = np.moveaxis(gt, 0, -1)
    pred = prediction.resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR)
    pred_arr = np.asarray(pred, dtype=np.float32) / 127.5 - 1.0

    gt_norm = np.linalg.norm(gt, axis=-1)
    pred_norm = np.linalg.norm(pred_arr, axis=-1)
    valid = np.isfinite(gt).all(axis=-1) & (gt_norm > 0.1) & (pred_norm > 0.1)
    gt_unit = gt / np.maximum(gt_norm[..., None], 1e-6)
    pred_unit = pred_arr / np.maximum(pred_norm[..., None], 1e-6)
    cosine = np.sum(gt_unit * pred_unit, axis=-1)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angles[valid].mean()), int(valid.sum())


def generated_image(result: dict, chained: bool) -> Image.Image:
    value = result["image"]
    if chained:
        assert isinstance(value, list) and len(value) >= 2
        value = value[-1]
    assert isinstance(value, Image.Image)
    return value


def image_call(inferencer, image: Image.Image, *, intermediate: str | None, seed: int, steps: int):
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = generate_any(
        inferencer,
        "/tmp",
        input_image=image,
        condition=["rgb"],
        target="normal",
        intermediate=intermediate,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        num_timesteps=steps,
        seed=seed,
        image_size=512,
        num_samples=1,
        use_instruction=False,
        use_target_instruction=False,
        use_condition_instruction=False,
        use_intermediate_instruction=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    return generated_image(results[0], chained=intermediate is not None), elapsed, peak


def auxiliary_calls(inferencer, image: Image.Image, rank: int) -> None:
    targets = ["depth", "canny"]
    if rank < 2:
        target = targets[rank]
        started = time.perf_counter()
        out = generate_any(
            inferencer,
            "/tmp",
            input_image=image,
            condition=["rgb"],
            target=target,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            num_timesteps=5,
            seed=SEED_BASE,
            image_size=512,
            num_samples=1,
            use_instruction=False,
            use_target_instruction=False,
            use_condition_instruction=False,
        )[0]["image"]
        torch.cuda.synchronize()
        emit({
            "claim": "shared_direct_mapping",
            "rank": rank,
            "condition": "rgb",
            "target": target,
            "output_type": type(out).__name__,
            "elapsed_seconds": time.perf_counter() - started,
        })
        return

    target = "caption" if rank == 2 else "det"
    decode_method = "text" if target == "caption" else "detection"
    hyper = _build_understanding_hyper(
        decode_method=decode_method,
        condition=["rgb"],
        target=target,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        use_instruction=False,
        use_target_instruction=False,
        use_condition_instruction=False,
        do_modality_norm=False,
        do_sample=False,
        temperature=0.003,
    )
    started = time.perf_counter()
    result = inferencer(
        image=image,
        text=("the largest object" if target == "det" else None),
        understanding_output=True,
        **hyper,
    )
    torch.cuda.synchronize()
    emit({
        "claim": "shared_direct_mapping",
        "rank": rank,
        "condition": ["rgb", "text"] if target == "det" else ["rgb"],
        "target": target,
        "output_type": type(result.get("text")).__name__,
        "output_preview": str(result.get("text"))[:240],
        "det_confidence": result.get("det_confidence"),
        "elapsed_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    dataset = Nyuv2Validation(args.dataset_dir)
    assert len(dataset) == 654

    load_started = time.perf_counter()
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
    emit({
        "claim": "one_checkpoint_shared_runtime",
        "rank": args.rank,
        "checkpoint": "EPFL-VILAB/MODUS",
        "model_class": type(model).__name__,
        "inferencer_class": type(inferencer).__name__,
        "registry_object_id": id(registry),
        "modality_count": len(registry.specs),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_seconds": time.perf_counter() - load_started,
    })

    assigned = SLICE_INDICES[args.rank :: args.world_size]
    records = []
    first_image = None
    for index in assigned:
        sample = dataset[index]
        image = as_rgb(sample["image"])
        first_image = first_image or image
        seed = SEED_BASE + index
        direct, direct_s, direct_peak = image_call(
            inferencer, image, intermediate=None, seed=seed, steps=args.steps
        )
        chain, chain_s, chain_peak = image_call(
            inferencer, image, intermediate="canny", seed=seed, steps=args.steps
        )
        direct_mae, valid_pixels = angular_mae(direct, sample["normal"])
        chain_mae, _ = angular_mae(chain, sample["normal"])
        record = {
            "claim": "nyuv2_canny_chain",
            "rank": args.rank,
            "index": index,
            "seed": seed,
            "steps": args.steps,
            "direct_mae_deg": direct_mae,
            "chain_mae_deg": chain_mae,
            "delta_chain_minus_direct_deg": chain_mae - direct_mae,
            "direct_seconds": direct_s,
            "chain_seconds": chain_s,
            "latency_added_seconds": chain_s - direct_s,
            "direct_peak_bytes": direct_peak,
            "chain_peak_bytes": chain_peak,
            "valid_pixels": valid_pixels,
        }
        records.append(record)
        emit(record)

    assert first_image is not None
    auxiliary_calls(inferencer, first_image, args.rank)
    emit({
        "claim": "nyuv2_rank_summary",
        "rank": args.rank,
        "n": len(records),
        "mean_direct_mae_deg": float(np.mean([r["direct_mae_deg"] for r in records])),
        "mean_chain_mae_deg": float(np.mean([r["chain_mae_deg"] for r in records])),
        "mean_delta_deg": float(np.mean([r["delta_chain_minus_direct_deg"] for r in records])),
        "mean_direct_seconds": float(np.mean([r["direct_seconds"] for r in records])),
        "mean_chain_seconds": float(np.mean([r["chain_seconds"] for r in records])),
        "max_peak_bytes": max(max(r["direct_peak_bytes"], r["chain_peak_bytes"]) for r in records),
    })


if __name__ == "__main__":
    main()
