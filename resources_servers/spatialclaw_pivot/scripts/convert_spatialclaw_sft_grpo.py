#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert SpatialClaw SFT trajectories to matched prompt-only GRPO rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from resources_servers.spatialclaw_pivot.action_utils import (
    canonicalize_spatialclaw_action,
)
from resources_servers.spatialclaw_pivot.scripts.convert_spatialclaw_sft_pivots import (
    _component_specs,
    _file_uri,
    _read_jsonl,
    _stable_id,
)


_IMAGE_PLACEHOLDER = "<image>"
_FORMAT = "spatialclaw_sft_prompt_grpo/v1"


def _task_instruction(conversations: Any) -> tuple[str, int]:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Missing non-empty conversations list")
    for message in conversations:
        if not isinstance(message, dict) or message.get("from") != "human":
            continue
        value = message.get("value")
        if not isinstance(value, str):
            raise ValueError("Initial human message has no string value")
        initial_key_frames = value.count(_IMAGE_PLACEHOLDER)
        if initial_key_frames not in {0, 256}:
            raise ValueError(
                "Expected either 0 or 256 initial key frames, found "
                f"{initial_key_frames}"
            )
        instruction = (
            value.rsplit(_IMAGE_PLACEHOLDER, 1)[-1].strip()
            if initial_key_frames
            else value.strip()
        )
        if not instruction:
            raise ValueError("Initial human message has no task after key frames")
        return instruction, initial_key_frames
    raise ValueError("Trajectory has no human task message")


def _terminal_answer(conversations: Any) -> dict[str, Any]:
    if not isinstance(conversations, list):
        raise ValueError("Missing conversations list")
    assistants = [
        message
        for message in conversations
        if isinstance(message, dict) and message.get("from") == "gpt"
    ]
    if not assistants:
        raise ValueError("Trajectory has no assistant actions")
    expected = canonicalize_spatialclaw_action(assistants[-1].get("value"))
    if expected is None or expected.get("type") != "final_answer":
        raise ValueError("Trajectory does not end with a final-answer action")
    return expected


def _video_path(media_root: Path, source_id: Any) -> Path:
    try:
        numeric_id = int(source_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Source row ID is not numeric: {source_id}") from exc
    return media_root / "videos" / f"{numeric_id:06d}.mp4"


def _convert_row(
    row: dict[str, Any],
    *,
    component: str,
    source_path: Path,
    line_number: int,
    media_root: Path,
    check_video: bool,
) -> dict[str, Any]:
    source_id = row.get("id", line_number)
    trajectory_id = _stable_id(component, source_id, line_number)
    instruction, initial_key_frames = _task_instruction(row.get("conversations"))
    terminal = _terminal_answer(row.get("conversations"))
    video_path = _video_path(media_root, source_id)
    if check_video and not video_path.is_file():
        raise ValueError(f"Missing source video: {video_path}")

    sample_id = f"grpo:{trajectory_id}"
    spatialclaw_metadata = {
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "source_component": component,
        "source_row_id": str(source_id),
        "source_line": line_number,
        "sft_initial_key_frames": initial_key_frames,
    }
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {
                            "type": "input_video",
                            "video_url": _file_uri(video_path),
                        },
                    ],
                }
            ],
            "metadata": {
                "nemo_rl_defer_multimodal_to_agent": "true",
                "spatialclaw": json.dumps(
                    spatialclaw_metadata, separators=(",", ":")
                ),
            },
        },
        "expected_answer": str(terminal.get("answer", "")),
        "scoring_mode": str(terminal.get("scoring_mode", "auto")),
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        # These are successful private SFT trajectories, not registered
        # SpatialClaw BenchmarkFactory datasets. A synthetic benchmark label
        # incorrectly routes `auto` scoring into the native benchmark loader.
        # Keep it null so the verifier scores the preserved terminal reference
        # with its deterministic MCQA/exact fallback.
        "benchmark": None,
        "source_name": component,
        "source_path": str(source_path),
        "source_row_id": source_id,
        "source_line": line_number,
        "video_path": str(video_path),
        "sft_initial_key_frames": initial_key_frames,
        "source_reasoning_mode": "non-reasoning-sft",
        "target_has_think_tags": False,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "spatialclaw_agent",
        },
    }


def convert(
    blend_path: Path,
    output_path: Path,
    *,
    limit_trajectories: int | None = None,
    check_video: bool = True,
    required_initial_key_frames: int | None = None,
) -> dict[str, Any]:
    if limit_trajectories is not None and limit_trajectories <= 0:
        raise ValueError("limit_trajectories must be positive")
    if required_initial_key_frames not in {None, 0, 256}:
        raise ValueError("required_initial_key_frames must be 0, 256, or None")

    specs = _component_specs(blend_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    hasher = hashlib.sha256()
    totals: Counter[str] = Counter()
    components: dict[str, Counter[str]] = {}
    rejections: Counter[str] = Counter()
    stopped = False

    try:
        with temporary_path.open("wb") as output:
            for spec in specs:
                component = str(spec["component"])
                counts: Counter[str] = Counter()
                components[component] = counts
                for line_number, row in _read_jsonl(spec["jsonl_path"]):
                    if (
                        limit_trajectories is not None
                        and totals["trajectories"] >= limit_trajectories
                    ):
                        stopped = True
                        break
                    try:
                        converted = _convert_row(
                            row,
                            component=component,
                            source_path=spec["jsonl_path"],
                            line_number=line_number,
                            media_root=spec["media_root"],
                            check_video=check_video,
                        )
                    except ValueError as exc:
                        reason = str(exc).split(":", 1)[0]
                        rejections[reason] += 1
                        counts["rejected_trajectories"] += 1
                        totals["rejected_trajectories"] += 1
                        continue

                    if (
                        required_initial_key_frames is not None
                        and converted["sft_initial_key_frames"]
                        != required_initial_key_frames
                    ):
                        counts["filtered_trajectories"] += 1
                        totals["filtered_trajectories"] += 1
                        continue

                    encoded = (
                        json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode()
                    output.write(encoded)
                    hasher.update(encoded)
                    totals["trajectories"] += 1
                    totals["prompts"] += 1
                    counts["trajectories"] += 1
                    counts["prompts"] += 1
                if stopped:
                    break
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": _FORMAT,
        "source_blend": str(blend_path.resolve()),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": hasher.hexdigest(),
        "input_media": "source video",
        "expert_history_in_prompt": False,
        "pivot_profiling": False,
        "sft_frame_setting": "validated trajectories with 0 or 256 initial key frames",
        "runtime_frame_setting": "launcher-controlled (32 or 256)",
        "check_video": check_video,
        "required_initial_key_frames": required_initial_key_frames,
        "limit_trajectories": limit_trajectories,
        "truncated": stopped,
        "components": {
            spec["component"]: {
                "component_manifest": str(spec["component_manifest"]),
                "jsonl_path": str(spec["jsonl_path"]),
                "media_root": str(spec["media_root"]),
                "subflavors": spec["subflavors"],
                "counts": dict(components.get(str(spec["component"]), {})),
            }
            for spec in specs
        },
        "totals": dict(totals),
        "rejections": dict(rejections),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_blend", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-trajectories", type=int)
    parser.add_argument("--skip-video-check", action="store_true")
    parser.add_argument(
        "--required-initial-key-frames",
        type=int,
        choices=(0, 256),
        help="Keep only trajectories whose SFT prefix has this frame count.",
    )
    args = parser.parse_args()
    manifest = convert(
        args.source_blend,
        args.output,
        limit_trajectories=args.limit_trajectories,
        check_video=not args.skip_video_check,
        required_initial_key_frames=args.required_initial_key_frames,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
