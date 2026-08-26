import json

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.spatialclaw_pivot.scripts.extract_spatialclaw_pivots import (
    extract,
)


def test_extract_builds_one_step_exact_history(tmp_path):
    workspace = tmp_path / "workspaces" / "session-a"
    workspace.mkdir(parents=True)
    capture = {
        "format": "spatialclaw_pivot_capture/v1",
        "session_id": "session-a",
        "video_input_mode": "key-frame-aware",
        "run_metadata": {"frame_indices": list(range(32))},
        "turns": [
            {
                "content": "x = vlm.ask(InputImages[0], question='what?')",
                "request_kwargs": {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "question"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "file:///tmp/frame.jpg"},
                                },
                            ],
                        },
                    ],
                    "extra_body": {"mm_processor_kwargs": {"video_as_images": True}},
                },
            }
        ],
    }
    (workspace / "rl_capture.json").write_text(json.dumps(capture))
    output = tmp_path / "pivots.jsonl"

    manifest = extract(
        workspace.parent,
        None,
        output,
        minimum_reward=1.0,
        include_unscored=True,
    )

    assert manifest["pivots"] == 1
    row = json.loads(output.read_text().strip())
    assert row["pivot_id"] == "session-a:0"
    assert row["expected_action"]["type"] == "python_calls"
    assert row["frame_indices"] == list(range(32))
    assert row["responses_create_params"]["metadata"]["pivot_id"] == "session-a:0"
    assert "tools" not in row["responses_create_params"]
    NeMoGymResponseCreateParamsNonStreaming.model_validate(
        row["responses_create_params"]
    )


def test_extract_maps_native_terminal_scoring_to_local_auto(tmp_path):
    workspace = tmp_path / "workspaces" / "session-a"
    workspace.mkdir(parents=True)
    capture = {
        "format": "spatialclaw_pivot_capture/v1",
        "session_id": "session-a",
        "turns": [
            {
                "content": "ReturnAnswer('model answer')",
                "request_kwargs": {
                    "messages": [{"role": "user", "content": "question"}]
                },
            }
        ],
    }
    (workspace / "rl_capture.json").write_text(json.dumps(capture))
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        json.dumps(
            {
                "reward": 1.0,
                "expected_answer": "B",
                "scoring_mode": "native",
                "response": {"metadata": {"spatialclaw_session_id": "session-a"}},
            }
        )
        + "\n"
    )
    output = tmp_path / "pivots.jsonl"

    extract(
        workspace.parent,
        rollouts,
        output,
        minimum_reward=1.0,
        include_unscored=False,
    )

    row = json.loads(output.read_text().strip())
    assert row["expected_action"] == {
        "type": "final_answer",
        "answer": "B",
        "scoring_mode": "auto",
    }


def test_extract_uses_verification_stored_with_capture(tmp_path):
    workspace = tmp_path / "workspaces" / "session-a"
    workspace.mkdir(parents=True)
    capture = {
        "format": "spatialclaw_pivot_capture/v1",
        "session_id": "session-a",
        "verification": {
            "reward": 1.0,
            "expected_answer": "C",
            "scoring_mode": "mcqa",
        },
        "turns": [
            {
                "content": "ReturnAnswer('C')",
                "request_kwargs": {
                    "messages": [{"role": "user", "content": "question"}]
                },
            }
        ],
    }
    (workspace / "rl_capture.json").write_text(json.dumps(capture))
    output = tmp_path / "pivots.jsonl"

    manifest = extract(
        workspace.parent,
        None,
        output,
        minimum_reward=1.0,
        include_unscored=False,
    )

    assert manifest["successful_trajectories"] == 1
    row = json.loads(output.read_text().strip())
    assert row["expected_action"] == {
        "type": "final_answer",
        "answer": "C",
        "scoring_mode": "mcqa",
    }


def test_extract_rejects_inexact_assistant_history(tmp_path):
    workspace = tmp_path / "workspaces" / "session-a"
    workspace.mkdir(parents=True)
    capture = {
        "format": "spatialclaw_pivot_capture/v1",
        "session_id": "session-a",
        "turns": [
            {
                "content": "ReturnAnswer('B')",
                "request_kwargs": {
                    "messages": [
                        {"role": "assistant", "content": "previous action"},
                        {"role": "user", "content": "tool feedback"},
                    ]
                },
            }
        ],
    }
    (workspace / "rl_capture.json").write_text(json.dumps(capture))

    with pytest.raises(ValueError, match="without exact prompt/generation"):
        extract(
            workspace.parent,
            None,
            tmp_path / "pivots.jsonl",
            minimum_reward=1.0,
            include_unscored=True,
        )
