#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert validated Energon SpatialClaw SFT trajectories to pivot candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from resources_servers.spatialclaw_pivot.action_utils import (
    canonicalize_spatialclaw_action,
)


_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}
_IMAGE_PLACEHOLDER = "<image>"
_FORMAT = "spatialclaw_sft_pivot_candidates/v1"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _blend_entries(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    try:
        entries = manifest["splits"]["train"]["blend_epochized"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Expected splits.train.blend_epochized in {path}"
        ) from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Expected a non-empty blend in {path}")
    return entries


def _resolve_path(value: str, parent: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _media_root(value: Any, path: Path) -> Path:
    if not isinstance(value, str) or not value.startswith("filesystem://"):
        raise ValueError(
            f"Expected aux.media_source with a filesystem:// URI in {path}"
        )
    root = Path(value.removeprefix("filesystem://"))
    if not root.is_absolute():
        raise ValueError(f"Expected an absolute media root in {path}: {root}")
    return root


def _component_specs(blend_path: Path) -> list[dict[str, Any]]:
    blend = _read_yaml(blend_path)
    specs: list[dict[str, Any]] = []
    for blend_entry in _blend_entries(blend, blend_path):
        component_manifest = _resolve_path(
            str(blend_entry["path"]), blend_path.parent
        )
        component = _read_yaml(component_manifest)
        entries = _blend_entries(component, component_manifest)
        if len(entries) != 1:
            raise ValueError(
                f"Expected exactly one JSONL entry in {component_manifest}"
            )
        entry = entries[0]
        jsonl_path = _resolve_path(
            str(entry["path"]), component_manifest.parent
        )
        specs.append(
            {
                "component": component_manifest.parent.name,
                "component_manifest": component_manifest,
                "jsonl_path": jsonl_path,
                "media_root": _media_root(
                    (entry.get("aux") or {}).get("media_source"),
                    component_manifest,
                ),
                "subflavors": copy.deepcopy(entry.get("subflavors") or {}),
            }
        )
    return specs


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            yield line_number, row


def _file_uri(path: Path) -> str:
    return path.absolute().as_uri()


def _message_with_media(
    role: str,
    text: str,
    images: list[str],
    image_cursor: int,
    media_root: Path,
) -> tuple[dict[str, Any], int]:
    placeholder_count = text.count(_IMAGE_PLACEHOLDER)
    if role == "assistant" and placeholder_count:
        raise ValueError("Assistant messages cannot introduce <image> parts")
    if placeholder_count == 0:
        return {"role": role, "content": text}, image_cursor
    if image_cursor + placeholder_count > len(images):
        raise ValueError("More <image> placeholders than row.image entries")

    content: list[dict[str, Any]] = []
    pieces = text.split(_IMAGE_PLACEHOLDER)
    for index, piece in enumerate(pieces):
        if piece:
            content.append({"type": "input_text", "text": piece})
        if index < placeholder_count:
            media_path = media_root / images[image_cursor]
            content.append(
                {
                    "type": "input_image",
                    "image_url": _file_uri(media_path),
                    "detail": "auto",
                }
            )
            image_cursor += 1
    return {"role": role, "content": content}, image_cursor


def _stable_id(component: str, source_id: Any, line_number: int) -> str:
    source = f"{component}\0{source_id!s}\0{line_number}".encode()
    digest = hashlib.sha256(source).hexdigest()[:16]
    return f"sft-corrected-v2:{component}:{source_id}:{digest}"


def _sft_media_request_metadata(
    *,
    initial_key_frames: int,
    prompt_image_count: int,
) -> tuple[dict[str, str], list[int], list[str]]:
    """Reproduce corrected-v2's preframed-video media boundaries.

    The SFT loader treats only the first user turn's 256 images as one video.
    Images returned by later tool observations remain ordinary images.  The
    vLLM frame-group extension carries that distinction without changing the
    preserved message history or dropping any media.
    """
    if initial_key_frames < 0 or prompt_image_count < initial_key_frames:
        raise ValueError(
            "Invalid SFT media counts: "
            f"initial={initial_key_frames}, prompt={prompt_image_count}"
        )
    if initial_key_frames <= 1:
        return {}, [1] * prompt_image_count, ["image"] * prompt_image_count

    ordinary_image_count = prompt_image_count - initial_key_frames
    frame_counts = [initial_key_frames] + [1] * ordinary_image_count
    group_types = ["video"] + ["image"] * ordinary_image_count
    extra_body = {
        "mm_processor_kwargs": {
            "max_num_tiles": 1,
            "video_as_images": True,
            "video_as_images_frame_counts": frame_counts,
            "video_as_images_group_types": group_types,
        }
    }
    return {
        "extra_body": json.dumps(extra_body, separators=(",", ":"))
    }, frame_counts, group_types


def _convert_row(
    row: dict[str, Any],
    *,
    component: str,
    source_path: Path,
    line_number: int,
    media_root: Path,
    check_media: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conversations = row.get("conversations")
    images = row.get("image")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Missing non-empty conversations list")
    if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
        raise ValueError("Missing string-valued image list")

    source_id = row.get("id", line_number)
    trajectory_id = _stable_id(component, source_id, line_number)
    prefix: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    image_cursor = 0
    decision_index = 0
    first_user_image_count: int | None = None

    for message_index, raw_message in enumerate(conversations):
        if not isinstance(raw_message, dict):
            raise ValueError(f"Conversation item {message_index} is not an object")
        source_role = raw_message.get("from")
        role = _ROLE_MAP.get(str(source_role))
        if role is None:
            raise ValueError(f"Unsupported role at message {message_index}: {source_role}")
        text = raw_message.get("value")
        if not isinstance(text, str):
            raise ValueError(f"Message {message_index} has no string value")

        if role == "assistant":
            expected_action = canonicalize_spatialclaw_action(text)
            if expected_action is None:
                raise ValueError(
                    f"Unsupported assistant action at decision {decision_index}"
                )
            pivot_id = f"{trajectory_id}:{decision_index}"
            metadata = {
                "pivot_id": pivot_id,
                "trajectory_id": trajectory_id,
                "source_component": component,
                "source_row_id": str(source_id),
                "source_line": str(line_number),
                "decision_index": str(decision_index),
            }
            media_metadata, media_group_counts, media_group_types = (
                _sft_media_request_metadata(
                    initial_key_frames=first_user_image_count or 0,
                    prompt_image_count=image_cursor,
                )
            )
            metadata.update(media_metadata)
            candidates.append(
                {
                    "responses_create_params": {
                        "input": copy.deepcopy(prefix),
                        "metadata": metadata,
                    },
                    "expected_action": expected_action,
                    "pivot_id": pivot_id,
                    "trajectory_id": trajectory_id,
                    "decision_index": decision_index,
                    "session_kind": "main",
                    "expert_source": "validated_spatialclaw_sft",
                    "expert_trajectory_reward": 1.0,
                    "video_input_mode": "key-frame-aware",
                    "num_key_frames": first_user_image_count,
                    "prompt_image_count": image_cursor,
                    "media_group_counts": media_group_counts,
                    "media_group_types": media_group_types,
                    "sft_preframed_images_as_video": bool(
                        first_user_image_count and first_user_image_count > 1
                    ),
                    "source": {
                        "component": component,
                        "jsonl_path": str(source_path),
                        "line_number": line_number,
                        "row_id": source_id,
                        "media_root": str(media_root),
                        "message_index": message_index,
                    },
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": "spatialclaw_pivot_agent",
                    },
                }
            )
            decision_index += 1

        message, next_image_cursor = _message_with_media(
            role, text, images, image_cursor, media_root
        )
        if role == "user" and first_user_image_count is None:
            first_user_image_count = next_image_cursor - image_cursor
        prefix.append(message)
        image_cursor = next_image_cursor

    if image_cursor != len(images):
        raise ValueError(
            f"Consumed {image_cursor} image references but row contains {len(images)}"
        )
    if not candidates:
        raise ValueError("Trajectory contains no supported assistant decisions")
    if candidates[-1]["expected_action"]["type"] != "final_answer":
        raise ValueError("Trajectory does not end with a final-answer action")
    if first_user_image_count not in {0, 256}:
        raise ValueError(
            "Expected either 0 or 256 initial key frames, found "
            f"{first_user_image_count}"
        )

    if check_media != "none" and images:
        paths = (
            [media_root / images[0], media_root / images[-1]]
            if check_media == "first-last"
            else [media_root / image for image in images]
        )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"Missing media files: {missing[:3]}")

    return candidates, {
        "assistant_turns": decision_index,
        "media_references": len(images),
        "initial_key_frames": first_user_image_count,
    }


def convert(
    blend_path: Path,
    output_path: Path,
    *,
    limit_trajectories: int | None = None,
    max_candidates: int | None = None,
    check_media: str = "first-last",
    required_initial_key_frames: int | None = None,
) -> dict[str, Any]:
    if limit_trajectories is not None and limit_trajectories <= 0:
        raise ValueError("limit_trajectories must be positive")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if check_media not in {"none", "first-last", "all"}:
        raise ValueError(f"Unsupported media check: {check_media}")
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
                        candidates, row_counts = _convert_row(
                            row,
                            component=component,
                            source_path=spec["jsonl_path"],
                            line_number=line_number,
                            media_root=spec["media_root"],
                            check_media=check_media,
                        )
                    except ValueError as exc:
                        reason = str(exc).split(":", 1)[0]
                        rejections[reason] += 1
                        counts["rejected_trajectories"] += 1
                        totals["rejected_trajectories"] += 1
                        continue

                    if (
                        required_initial_key_frames is not None
                        and row_counts["initial_key_frames"]
                        != required_initial_key_frames
                    ):
                        totals["filtered_trajectories"] += 1
                        counts["filtered_trajectories"] += 1
                        continue

                    totals["trajectories"] += 1
                    counts["trajectories"] += 1
                    for key, value in row_counts.items():
                        totals[key] += int(value)
                        counts[key] += int(value)
                    for candidate in candidates:
                        if (
                            max_candidates is not None
                            and totals["candidates"] >= max_candidates
                        ):
                            stopped = True
                            break
                        encoded = (
                            json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        ).encode()
                        output.write(encoded)
                        hasher.update(encoded)
                        totals["candidates"] += 1
                        counts["candidates"] += 1
                        action_type = candidate["expected_action"]["type"]
                        totals[f"action_{action_type}"] += 1
                        counts[f"action_{action_type}"] += 1
                    if stopped:
                        break
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
        "frame_setting": "validated SFT prefix (0 or 256 initial key frames)",
        "media_semantics": (
            "first user image block is one preframed video; later tool "
            "observation images are independent images"
        ),
        "history_tokenization": "runtime policy tokenizer; no fabricated token IDs",
        "check_media": check_media,
        "limit_trajectories": limit_trajectories,
        "max_candidates": max_candidates,
        "required_initial_key_frames": required_initial_key_frames,
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
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--required-initial-key-frames",
        type=int,
        choices=(0, 256),
        help="Keep only trajectories whose SFT prefix has this frame count.",
    )
    parser.add_argument(
        "--check-media",
        choices=("none", "first-last", "all"),
        default="first-last",
    )
    args = parser.parse_args()
    manifest = convert(
        args.source_blend,
        args.output,
        limit_trajectories=args.limit_trajectories,
        max_candidates=args.max_candidates,
        check_media=args.check_media,
        required_initial_key_frames=args.required_initial_key_frames,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
