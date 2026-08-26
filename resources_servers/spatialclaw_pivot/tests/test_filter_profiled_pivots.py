import json
from pathlib import Path

from resources_servers.spatialclaw_pivot.scripts.filter_profiled_pivots import (
    filter_profiled_pivots,
)


def _sample(pivot_id: str, image_count: int) -> dict:
    return {
        "pivot_id": pivot_id,
        "responses_create_params": {
            "seed": 0,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"file:///frame-{index}.png",
                        }
                        for index in range(image_count)
                    ],
                },
                {"role": "assistant", "content": "show(InputImages[0])"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "file:///later-tool-result.png",
                        }
                    ],
                },
            ]
        },
    }


def _write_profiles(path: Path) -> None:
    rows = [
        {
            "mean/reward": 0.5,
            "std/reward": 0.5,
            "sample": _sample("keep", 256),
        },
        {
            "mean/reward": 0.8,
            "std/reward": 0.4,
            "sample": _sample("too-easy", 256),
        },
        {
            "mean/reward": 0.0,
            "std/reward": 0.0,
            "sample": _sample("no-variance", 256),
        },
        {
            "mean/reward": 0.5,
            "std/reward": 0.5,
            "sample": _sample("wrong-frame-pool", 32),
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_filter_profiled_pivots_applies_variance_difficulty_and_frame_gates(
    tmp_path,
):
    profiles = tmp_path / "profiles.jsonl"
    selected = tmp_path / "selected.jsonl"
    _write_profiles(profiles)

    manifest = filter_profiled_pivots(
        profiles,
        selected,
        difficulty_threshold=0.75,
        required_initial_key_frames=256,
    )

    rows = [json.loads(line) for line in selected.read_text().splitlines()]
    assert [row["pivot_id"] for row in rows] == ["keep"]
    assert rows[0]["pivot_profile"] == {
        "reward_mean": 0.5,
        "reward_std": 0.5,
        "difficulty_threshold": 0.75,
        "minimum_std": 0.0,
    }
    assert "seed" not in rows[0]["responses_create_params"]
    assert manifest["candidates"] == 4
    assert manifest["frame_filtered"] == 1
    assert manifest["profiling_seeds_removed"] == 1
    assert manifest["selected"] == 1
    assert manifest["required_initial_key_frames"] == 256


def test_filter_profiled_pivots_accepts_32_frame_gate(tmp_path):
    profiles = tmp_path / "profiles.jsonl"
    selected = tmp_path / "selected.jsonl"
    profiles.write_text(
        json.dumps(
            {
                "mean/reward": 0.5,
                "std/reward": 0.5,
                "sample": _sample("keep-32", 32),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = filter_profiled_pivots(
        profiles,
        selected,
        difficulty_threshold=0.75,
        required_initial_key_frames=32,
    )

    assert manifest["selected"] == 1
    assert manifest["required_initial_key_frames"] == 32
