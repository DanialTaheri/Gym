import json
from pathlib import Path

import yaml

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.spatialclaw_pivot.scripts.convert_spatialclaw_sft_grpo import (
    convert as convert_grpo,
)
from resources_servers.spatialclaw_pivot.scripts.convert_spatialclaw_sft_pivots import (
    convert,
)


def _write_fixture(tmp_path: Path, initial_key_frames: int = 256) -> Path:
    media_root = tmp_path / "media"
    media_root.mkdir()
    image_names = [
        f"images/{index:04d}.png" for index in range(initial_key_frames + 1)
    ]
    for image_name in image_names:
        image_path = media_root / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
    video_path = media_root / "videos" / "000017.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")

    component_root = tmp_path / "attribute"
    component_root.mkdir()
    row = {
        "id": 17,
        "conversations": [
            {"from": "system", "value": "SpatialClaw system"},
            {
                "from": "human",
                "value": (
                    (
                        "SFT key-frame overview\n"
                        + "\n".join(["<image>"] * initial_key_frames)
                        + "\n"
                        if initial_key_frames
                        else ""
                    )
                    + "Which option is correct?\nA. No\nB. Yes"
                ),
            },
            {
                "from": "gpt",
                "value": "**Code**:\n```python\nshow(InputImages[0])\n```",
            },
            {
                "from": "human",
                "value": "Execution succeeded.\n<image>",
            },
            {
                "from": "gpt",
                "value": "**Code**:\n```python\nReturnAnswer('B')\n```",
            },
        ],
        "image": image_names,
    }
    jsonl_path = component_root / "energon_sft.jsonl"
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    component_manifest = component_root / "dataset.yaml"
    component_manifest.write_text(
        yaml.safe_dump(
            {
                "splits": {
                    "train": {
                        "blend_epochized": [
                            {
                                "path": str(jsonl_path),
                                "subflavors": {"name": "attribute"},
                                "aux": {
                                    "media_source": f"filesystem://{media_root}"
                                },
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    blend_path = tmp_path / "blend.yaml"
    blend_path.write_text(
        yaml.safe_dump(
            {
                "splits": {
                    "train": {
                        "blend_epochized": [{"path": str(component_manifest)}]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return blend_path


def _contains_token_metadata(value):
    if isinstance(value, dict):
        if {"prompt_token_ids", "generation_token_ids", "generation_log_probs"} & set(
            value
        ):
            return True
        return any(_contains_token_metadata(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_token_metadata(item) for item in value)
    return False


def test_convert_sft_builds_every_assistant_boundary(tmp_path):
    blend_path = _write_fixture(tmp_path)
    output_path = tmp_path / "pivots.jsonl"

    manifest = convert(blend_path, output_path, check_media="all")

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert manifest["totals"]["trajectories"] == 1
    assert manifest["totals"]["candidates"] == 2
    assert manifest["totals"]["action_python_calls"] == 1
    assert manifest["totals"]["action_final_answer"] == 1
    assert manifest["truncated"] is False
    assert len(rows) == 2

    tool_candidate, final_candidate = rows
    assert tool_candidate["decision_index"] == 0
    assert tool_candidate["num_key_frames"] == 256
    assert tool_candidate["prompt_image_count"] == 256
    assert tool_candidate["expected_action"]["type"] == "python_calls"
    assert [call["name"] for call in tool_candidate["expected_action"]["calls"]] == [
        "show"
    ]
    assert [message["role"] for message in tool_candidate["responses_create_params"]["input"]] == [
        "system",
        "user",
    ]
    tool_extra_body = json.loads(
        tool_candidate["responses_create_params"]["metadata"]["extra_body"]
    )
    assert tool_extra_body == {
        "mm_processor_kwargs": {
            "max_num_tiles": 1,
            "video_as_images": True,
            "video_as_images_frame_counts": [256],
            "video_as_images_group_types": ["video"],
        }
    }
    assert tool_candidate["media_group_counts"] == [256]
    assert tool_candidate["media_group_types"] == ["video"]

    assert final_candidate["decision_index"] == 1
    assert final_candidate["prompt_image_count"] == 257
    assert final_candidate["expected_action"] == {
        "type": "final_answer",
        "answer": "B",
        "scoring_mode": "auto",
    }
    assert [message["role"] for message in final_candidate["responses_create_params"]["input"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    final_extra_body = json.loads(
        final_candidate["responses_create_params"]["metadata"]["extra_body"]
    )
    assert final_extra_body["mm_processor_kwargs"][
        "video_as_images_frame_counts"
    ] == [256, 1]
    assert final_extra_body["mm_processor_kwargs"][
        "video_as_images_group_types"
    ] == ["video", "image"]
    assert final_candidate["media_group_counts"] == [256, 1]
    assert final_candidate["media_group_types"] == ["video", "image"]

    assert not _contains_token_metadata(rows)
    for row in rows:
        NeMoGymResponseCreateParamsNonStreaming.model_validate(
            row["responses_create_params"]
        )

    image_parts = [
        part
        for message in final_candidate["responses_create_params"]["input"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "input_image"
    ]
    assert len(image_parts) == 257
    assert image_parts[0]["image_url"].endswith("/images/0000.png")
    assert image_parts[-1]["image_url"].endswith("/images/0256.png")

    manifest_on_disk = json.loads(
        output_path.with_suffix(".manifest.json").read_text()
    )
    assert manifest_on_disk["output_sha256"] == manifest["output_sha256"]


def test_convert_sft_candidate_limit_is_deterministic(tmp_path):
    blend_path = _write_fixture(tmp_path)
    output_path = tmp_path / "one-pivot.jsonl"

    manifest = convert(
        blend_path,
        output_path,
        max_candidates=1,
        check_media="first-last",
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert manifest["truncated"] is True
    assert manifest["totals"]["candidates"] == 1
    assert rows[0]["decision_index"] == 0


def test_convert_sft_grpo_keeps_only_task_video_and_reference(tmp_path):
    blend_path = _write_fixture(tmp_path)
    output_path = tmp_path / "grpo.jsonl"

    manifest = convert_grpo(blend_path, output_path, check_video=True)

    row = json.loads(output_path.read_text())
    assert manifest["totals"] == {"trajectories": 1, "prompts": 1}
    assert manifest["expert_history_in_prompt"] is False
    assert manifest["pivot_profiling"] is False
    assert row["expected_answer"] == "B"
    assert row["scoring_mode"] == "auto"
    assert row["benchmark"] is None
    assert row["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "spatialclaw_agent",
    }

    params = row["responses_create_params"]
    assert params["metadata"]["nemo_rl_defer_multimodal_to_agent"] == "true"
    metadata = json.loads(params["metadata"]["spatialclaw"])
    assert metadata["source_component"] == "attribute"
    assert metadata["source_row_id"] == "17"
    assert len(params["input"]) == 1
    assert params["input"][0]["role"] == "user"
    text_part, video_part = params["input"][0]["content"]
    assert text_part == {
        "type": "input_text",
        "text": "Which option is correct?\nA. No\nB. Yes",
    }
    assert video_part["type"] == "input_video"
    assert video_part["video_url"].endswith("/media/videos/000017.mp4")
    assert "show(" not in json.dumps(params)
    assert "ReturnAnswer" not in json.dumps(params)
    NeMoGymResponseCreateParamsNonStreaming.model_validate(params)


def test_convert_sft_preserves_valid_zero_overview_trajectory(tmp_path):
    blend_path = _write_fixture(tmp_path, initial_key_frames=0)
    pivot_path = tmp_path / "zero-overview-pivots.jsonl"
    grpo_path = tmp_path / "zero-overview-grpo.jsonl"

    pivot_manifest = convert(blend_path, pivot_path, check_media="all")
    grpo_manifest = convert_grpo(blend_path, grpo_path, check_video=True)

    pivots = [json.loads(line) for line in pivot_path.read_text().splitlines()]
    assert pivot_manifest["rejections"] == {}
    assert [row["num_key_frames"] for row in pivots] == [0, 0]
    assert [row["prompt_image_count"] for row in pivots] == [0, 1]
    assert all(
        "extra_body" not in row["responses_create_params"]["metadata"]
        for row in pivots
    )
    assert [row["media_group_types"] for row in pivots] == [[], ["image"]]
    assert grpo_manifest["rejections"] == {}
    grpo = json.loads(grpo_path.read_text())
    assert grpo["responses_create_params"]["input"][0]["content"][0]["text"] == (
        "Which option is correct?\nA. No\nB. Yes"
    )


def test_convert_sft_can_filter_to_strict_256_frame_trajectories(tmp_path):
    blend_path = _write_fixture(tmp_path, initial_key_frames=0)
    pivot_path = tmp_path / "strict-pivots.jsonl"
    grpo_path = tmp_path / "strict-grpo.jsonl"

    pivot_manifest = convert(
        blend_path,
        pivot_path,
        check_media="all",
        required_initial_key_frames=256,
    )
    grpo_manifest = convert_grpo(
        blend_path,
        grpo_path,
        check_video=True,
        required_initial_key_frames=256,
    )

    assert pivot_path.read_text() == ""
    assert grpo_path.read_text() == ""
    assert pivot_manifest["totals"] == {"filtered_trajectories": 1}
    assert grpo_manifest["totals"] == {"filtered_trajectories": 1}
    assert pivot_manifest["required_initial_key_frames"] == 256
    assert grpo_manifest["required_initial_key_frames"] == 256
