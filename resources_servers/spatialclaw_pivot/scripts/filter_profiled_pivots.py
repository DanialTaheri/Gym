#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the PivotRL variance and difficulty gates to Gym reward profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _initial_image_count(sample: dict) -> int:
    """Count immutable task images before the first expert assistant action."""
    request = sample.get("responses_create_params")
    messages = request.get("input") if isinstance(request, dict) else None
    if not isinstance(messages, list):
        raise ValueError("sample has no responses_create_params.input messages")
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            break
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in content
        )
    return count


def filter_profiled_pivots(
    profiled_jsonl: Path,
    output_jsonl: Path,
    *,
    difficulty_threshold: float,
    minimum_std: float = 0.0,
    required_initial_key_frames: int | None = None,
) -> dict:
    if required_initial_key_frames is not None and required_initial_key_frames < 0:
        raise ValueError("required_initial_key_frames must be non-negative or None")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_jsonl.with_name(f".{output_jsonl.name}.tmp.{os.getpid()}")
    seen = kept = frame_filtered = profiling_seeds_removed = 0
    try:
        source = profiled_jsonl.open(encoding="utf-8")
        output = temporary_path.open("w", encoding="utf-8")
        with source, output:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                seen += 1
                mean = float(row.get("mean/reward", row.get("avg_reward", 0.0)))
                std = float(row.get("std/reward", row.get("std_reward", 0.0)))
                if std <= minimum_std or mean >= difficulty_threshold:
                    continue
                sample = row.get("sample")
                if not isinstance(sample, dict):
                    raise ValueError(f"Profile row {line_number} has no sample object")
                if (
                    required_initial_key_frames is not None
                    and _initial_image_count(sample) != required_initial_key_frames
                ):
                    frame_filtered += 1
                    continue
                sample = dict(sample)
                # ``ng_collect_rollouts +num_repeats_add_seed=true`` materializes
                # each profiling repeat with a fixed request seed.  The reward
                # profile retains one of those materialized samples (normally
                # seed 0).  That seed is profiling provenance, not part of the
                # pivot state: retaining it would make every online GRPO
                # generation for this pivot token-identical.
                request = sample.get("responses_create_params")
                if isinstance(request, dict) and "seed" in request:
                    request = dict(request)
                    request.pop("seed")
                    sample["responses_create_params"] = request
                    profiling_seeds_removed += 1
                sample["pivot_profile"] = {
                    "reward_mean": mean,
                    "reward_std": std,
                    "difficulty_threshold": difficulty_threshold,
                    "minimum_std": minimum_std,
                }
                output.write(json.dumps(sample, separators=(",", ":")) + "\n")
                kept += 1
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(output_jsonl)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "spatialclaw_profiled_pivot_dataset/v1",
        "profiled_jsonl": str(profiled_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "difficulty_threshold": difficulty_threshold,
        "minimum_std": minimum_std,
        "required_initial_key_frames": required_initial_key_frames,
        "candidates": seen,
        "frame_filtered": frame_filtered,
        "profiling_seeds_removed": profiling_seeds_removed,
        "selected": kept,
    }
    output_jsonl.with_suffix(".manifest.yaml").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiled_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--difficulty-threshold", type=float, required=True)
    parser.add_argument("--minimum-std", type=float, default=0.0)
    parser.add_argument("--required-initial-key-frames", type=int)
    args = parser.parse_args()

    manifest = filter_profiled_pivots(
        args.profiled_jsonl,
        args.output_jsonl,
        difficulty_threshold=args.difficulty_threshold,
        minimum_std=args.minimum_std,
        required_initial_key_frames=args.required_initial_key_frames,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
