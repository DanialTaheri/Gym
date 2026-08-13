#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert successful SpatialClaw capture workspaces into one-step pivots."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from resources_servers.spatialclaw_pivot.action_utils import (
    canonicalize_spatialclaw_action,
)


_SUPPORTED_SCORING_MODES = {"auto", "mcqa", "exact", "token_f1"}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def _response_metadata(row: dict[str, Any]) -> dict[str, Any]:
    for candidate in (
        row.get("response"),
        (row.get("full_result") or {}).get("response"),
        row.get("full_result"),
    ):
        if isinstance(candidate, dict) and isinstance(candidate.get("metadata"), dict):
            return candidate["metadata"]
    return {}


def _rollout_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        session_id = str(_response_metadata(row).get("spatialclaw_session_id") or "")
        if not session_id:
            continue
        reward = float(row.get("reward", (row.get("full_result") or {}).get("reward", 0.0)))
        previous = indexed.get(session_id)
        if previous is None or reward > float(previous["reward"]):
            indexed[session_id] = {"reward": reward, "row": row}
    return indexed


def _input_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    result = copy.deepcopy(part)
    part_type = result.get("type")
    if part_type == "text":
        result["type"] = "input_text"
    elif part_type in {"image", "image_url"}:
        result["type"] = "input_image"
    elif part_type in {"video", "video_url"}:
        result["type"] = "input_video"
    return result


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        message = copy.deepcopy(raw)
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if role == "assistant":
            text = content if isinstance(content, str) else ""
            item: dict[str, Any] = {
                "id": f"pivot-history-{index}",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
            for field in (
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
            ):
                if field in message:
                    item[field] = message[field]
            if all(
                field in item
                for field in (
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                )
            ):
                result.append(item)
            else:
                raise ValueError(
                    "SpatialClaw pivot history contains an assistant turn "
                    "without exact prompt/generation token metadata"
                )
            continue

        normalized_content = (
            [_input_part(part) for part in content]
            if isinstance(content, list)
            else content
        )
        result.append(
            {
                "role": role if role in {"system", "developer", "user"} else "user",
                "type": "message",
                "content": normalized_content,
            }
        )
    return result


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        tool = copy.deepcopy(tool)
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append({"type": "function"} | tool["function"])
        else:
            result.append(tool)
    return result


def _responses_create_params(
    request_kwargs: dict[str, Any], pivot_id: str
) -> dict[str, Any]:
    tools = _responses_tools(request_kwargs.get("tools") or [])
    params: dict[str, Any] = {
        "input": _responses_input(request_kwargs.get("messages") or []),
        "metadata": {"pivot_id": pivot_id},
    }
    if tools:
        params["parallel_tool_calls"] = False
        params["tool_choice"] = request_kwargs.get("tool_choice") or "auto"
        params["tools"] = tools
    if request_kwargs.get("max_tokens") is not None:
        params["max_output_tokens"] = int(request_kwargs["max_tokens"])
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    extra_body = copy.deepcopy(request_kwargs.get("extra_body") or {})
    if extra_body:
        params["metadata"]["extra_body"] = json.dumps(extra_body, separators=(",", ":"))
    return params


def extract(
    workspace_root: Path,
    rollout_path: Path | None,
    output_path: Path,
    *,
    minimum_reward: float,
    include_unscored: bool,
) -> dict[str, Any]:
    rollout_by_session = _rollout_index(rollout_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "captures": 0,
        "successful_trajectories": 0,
        "pivots": 0,
        "unsupported_actions": 0,
        "unscored_trajectories": 0,
    }
    with output_path.open("w", encoding="utf-8") as output:
        for capture_path in sorted(workspace_root.glob("*/rl_capture.json")):
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            if capture.get("format") != "spatialclaw_pivot_capture/v1":
                continue
            counts["captures"] += 1
            session_id = str(capture.get("session_id") or capture_path.parent.name)
            rollout = rollout_by_session.get(session_id)
            verification = capture.get("verification")
            if rollout is None and isinstance(verification, dict) and (
                "reward" in verification
            ):
                rollout = {
                    "reward": float(verification["reward"]),
                    "row": verification,
                }
            if rollout is None:
                counts["unscored_trajectories"] += 1
                if not include_unscored:
                    continue
            elif float(rollout["reward"]) < minimum_reward:
                continue
            counts["successful_trajectories"] += 1

            source_row = (rollout or {}).get("row") or {}
            for decision_index, turn in enumerate(capture.get("turns") or []):
                expected = canonicalize_spatialclaw_action(turn.get("content", ""))
                if expected is None:
                    counts["unsupported_actions"] += 1
                    continue
                if expected["type"] == "final_answer" and source_row.get("expected_answer"):
                    expected["answer"] = str(source_row["expected_answer"])
                    scoring_mode = str(source_row.get("scoring_mode") or "auto")
                    # SpatialClaw's online resource supports benchmark-native
                    # scorers. A one-step pivot has no benchmark server state,
                    # so use the equivalent local auto scorer for those rows.
                    expected["scoring_mode"] = (
                        scoring_mode
                        if scoring_mode in _SUPPORTED_SCORING_MODES
                        else "auto"
                    )
                pivot_id = f"{session_id}:{decision_index}"
                row = {
                    "responses_create_params": _responses_create_params(
                        turn.get("request_kwargs") or {}, pivot_id
                    ),
                    "expected_action": expected,
                    "pivot_id": pivot_id,
                    "trajectory_id": session_id,
                    "decision_index": decision_index,
                    "session_kind": "main",
                    "video_input_mode": capture.get("video_input_mode", "key-frame-aware"),
                    "frame_indices": (capture.get("run_metadata") or {}).get("frame_indices", []),
                    "expert_trajectory_reward": float((rollout or {}).get("reward", 0.0)),
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": "spatialclaw_pivot_agent",
                    },
                }
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts["pivots"] += 1

    manifest = {
        "format": "spatialclaw_pivot_dataset/v1",
        "workspace_root": str(workspace_root.resolve()),
        "rollouts": str(rollout_path.resolve()) if rollout_path else None,
        "output": str(output_path.resolve()),
        "minimum_expert_trajectory_reward": minimum_reward,
        "include_unscored": include_unscored,
        **counts,
    }
    output_path.with_suffix(".manifest.yaml").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--minimum-reward", type=float, default=1.0)
    parser.add_argument("--include-unscored", action="store_true")
    args = parser.parse_args()
    manifest = extract(
        args.workspace_root,
        args.rollouts,
        args.output,
        minimum_reward=args.minimum_reward,
        include_unscored=args.include_unscored,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
