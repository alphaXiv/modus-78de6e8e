#!/usr/bin/env python3
"""Bounded GenEval best-of-four test using MODUS grounding likelihood."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from any2any.any2any_tasks import (
    _build_understanding_hyper,
    create_inferencer,
    generate_any,
)
from any2any.load_any2any import load_any2any_model_hf


# Exact rows from GenEval commit af4902f, balanced across the two object-only
# categories. These require no color classifier and keep the external evaluator
# bounded to COCO object presence.
PROMPTS = [
    {"index": 0, "tag": "single_object", "prompt": "a photo of a bench", "objects": ["bench"]},
    {"index": 13, "tag": "single_object", "prompt": "a photo of a zebra", "objects": ["zebra"]},
    {"index": 52, "tag": "single_object", "prompt": "a photo of a banana", "objects": ["banana"]},
    {"index": 78, "tag": "single_object", "prompt": "a photo of a bear", "objects": ["bear"]},
    {"index": 104, "tag": "two_object", "prompt": "a photo of a zebra and a bed", "objects": ["zebra", "bed"]},
    {"index": 118, "tag": "two_object", "prompt": "a photo of a person and a bear", "objects": ["person", "bear"]},
    {"index": 129, "tag": "two_object", "prompt": "a photo of a chair and a bench", "objects": ["chair", "bench"]},
    {"index": 136, "tag": "two_object", "prompt": "a photo of a cow and a horse", "objects": ["cow", "horse"]},
]
SEED_BASE = 260725948


def emit(payload: dict) -> None:
    print("ORX_EVIDENCE " + json.dumps(payload, sort_keys=True), flush=True)


def generate_candidate(inferencer, prompt: str, seed: int) -> tuple[Image.Image, float]:
    started = time.perf_counter()
    result = generate_any(
        inferencer,
        "/tmp",
        prompt=prompt,
        condition=["caption"],
        target="rgb",
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        num_timesteps=10,
        seed=seed,
        image_size=512,
        num_samples=1,
        use_instruction=False,
        use_target_instruction=False,
        use_condition_instruction=False,
    )[0]
    torch.cuda.synchronize()
    assert isinstance(result["image"], Image.Image)
    return result["image"], time.perf_counter() - started


def grounding_score(inferencer, image: Image.Image, objects: list[str]) -> tuple[float, list[dict]]:
    scores = []
    details = []
    for object_name in objects:
        hyper = _build_understanding_hyper(
            decode_method="detection",
            condition=["rgb"],
            target="det",
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            use_instruction=False,
            use_target_instruction=False,
            use_condition_instruction=False,
            do_modality_norm=False,
            do_sample=False,
            temperature=0.003,
        )
        result = inferencer(
            image=image,
            text=object_name,
            understanding_output=True,
            **hyper,
        )
        confidence = result.get("det_confidence")
        numeric = float(confidence) if confidence is not None else 0.0
        scores.append(numeric)
        details.append({
            "object": object_name,
            "confidence": numeric,
            "output": str(result.get("text"))[:160],
        })
    return min(scores), details


def edge_density(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float((cv2.Canny(gray, 100, 200) > 0).mean())


class CocoDetrEvaluator:
    def __init__(self):
        model_id = "facebook/detr-resnet-50"
        self.processor = DetrImageProcessor.from_pretrained(model_id, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_id, revision="no_timm").eval().cuda()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def score(self, image: Image.Image, required: list[str]) -> tuple[int, dict]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device="cuda")
        detections = self.processor.post_process_object_detection(
            outputs, threshold=0.35, target_sizes=target_sizes
        )[0]
        labels = [self.id2label[int(i)] for i in detections["labels"].tolist()]
        passed = int(all(name in labels for name in required))
        return passed, {
            "detected": labels,
            "max_score": float(detections["scores"].max().item()) if len(labels) else 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
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
    evaluator = CocoDetrEvaluator()
    emit({
        "claim": "selfverify_shared_runtime",
        "rank": args.rank,
        "checkpoint": "EPFL-VILAB/MODUS",
        "model_class": type(model).__name__,
        "inferencer_class": type(inferencer).__name__,
        "evaluator": "facebook/detr-resnet-50@no_timm",
        "load_seconds": time.perf_counter() - load_started,
    })

    records = []
    for spec in PROMPTS[args.rank :: args.world_size]:
        candidates = []
        for candidate_index in range(4):
            seed = SEED_BASE + spec["index"] * 10 + candidate_index
            image, generation_seconds = generate_candidate(inferencer, spec["prompt"], seed)
            verifier, verifier_details = grounding_score(inferencer, image, spec["objects"])
            objective, objective_details = evaluator.score(image, spec["objects"])
            candidates.append({
                "candidate": candidate_index,
                "seed": seed,
                "generation_seconds": generation_seconds,
                "verifier_score": verifier,
                "verifier_details": verifier_details,
                "edge_density": edge_density(image),
                "objective_pass": objective,
                "objective_details": objective_details,
            })

        self_index = int(np.argmax([c["verifier_score"] for c in candidates]))
        edge_index = int(np.argmax([c["edge_density"] for c in candidates]))
        fixed_random_index = 0
        outcomes = [c["objective_pass"] for c in candidates]
        record = {
            "claim": "geneval_best_of_4",
            "rank": args.rank,
            **spec,
            "candidates": candidates,
            "self_selected_index": self_index,
            "edge_selected_index": edge_index,
            "fixed_random_index": fixed_random_index,
            "self_selected_pass": outcomes[self_index],
            "edge_selected_pass": outcomes[edge_index],
            "fixed_random_pass": outcomes[fixed_random_index],
            "expected_random_pass": float(np.mean(outcomes)),
            "oracle_pass": int(max(outcomes)),
            "self_minus_expected_random": outcomes[self_index] - float(np.mean(outcomes)),
        }
        records.append(record)
        emit(record)

    emit({
        "claim": "geneval_rank_summary",
        "rank": args.rank,
        "n_prompts": len(records),
        "self_pass_rate": float(np.mean([r["self_selected_pass"] for r in records])),
        "edge_pass_rate": float(np.mean([r["edge_selected_pass"] for r in records])),
        "fixed_random_pass_rate": float(np.mean([r["fixed_random_pass"] for r in records])),
        "expected_random_pass_rate": float(np.mean([r["expected_random_pass"] for r in records])),
        "oracle_pass_rate": float(np.mean([r["oracle_pass"] for r in records])),
    })


if __name__ == "__main__":
    main()
