from types import SimpleNamespace

import pytest

from responses_api_agents.spatialclaw_agent.app import (
    SpatialClawAgentConfig,
    _configure_compact_main_prompt,
    _configure_compact_planner_prompt,
    _configure_reasoning_roles,
    _configure_video_role_preprocessing,
    _finish_timed_out_capture,
    _minimum_frame_retry_fps,
    _session_id,
)


def test_key_frame_aware_is_the_default_video_input_mode():
    assert (
        SpatialClawAgentConfig.model_fields["video_input_mode"].default
        == "key-frame-aware"
    )


def test_compact_main_preserves_existing_prompt_ablations():
    config = SimpleNamespace(
        prompt_section_ablations={
            "exclude": ["reflection_checklist"],
            "override": {"header": "/tmp/custom-header.md"},
        }
    )

    _configure_compact_main_prompt(config)
    _configure_compact_main_prompt(config)

    assert config.prompt_section_ablations == {
        "exclude": ["reflection_checklist", "available_tools"],
        "override": {"header": "/tmp/custom-header.md"},
    }


def test_compact_planner_preserves_existing_prompt_ablations():
    config = SimpleNamespace(
        prompt_section_ablations={
            "exclude": ["reflection_checklist"],
            "override": {"planning_header": "/tmp/custom-header.md"},
        }
    )

    _configure_compact_planner_prompt(config)
    _configure_compact_planner_prompt(config)

    assert config.prompt_section_ablations == {
        "exclude": ["reflection_checklist", "planning_available_tools"],
        "override": {"planning_header": "/tmp/custom-header.md"},
    }


def test_session_id_accepts_filename_safe_value():
    assert _session_id("rollout-12.example") == "rollout-12.example"


def test_session_id_generates_uuid_when_value_is_empty():
    assert len(_session_id("")) == 32


def test_timed_out_capture_keeps_real_turns_and_uses_empty_answer(monkeypatch):
    captured_turn = object()
    monkeypatch.setattr(
        "responses_api_agents.spatialclaw_agent.app.finish_capture",
        lambda session_id: [captured_turn],
    )

    turns, result = _finish_timed_out_capture("rollout-12")

    assert turns == [captured_turn]
    assert result == {
        "final_answer": {"text": ""},
        "termination_reason": "timeout",
    }


@pytest.mark.parametrize("value", ["../escape", "/tmp/escape", "has space"])
def test_session_id_rejects_unsafe_explicit_value(value):
    with pytest.raises(ValueError, match="filename-safe"):
        _session_id(value)


def test_minimum_frame_retry_fps_raises_short_video_extraction_rate():
    retry = _minimum_frame_retry_fps(
        extracted_count=23,
        native_fps=29.88,
        total_native_frames=348,
        minimum_frames=32,
    )

    assert retry is not None
    assert 2.9 < retry < 3.0


def test_minimum_frame_retry_fps_skips_sufficient_or_too_short_video():
    assert (
        _minimum_frame_retry_fps(
            extracted_count=32,
            native_fps=30.0,
            total_native_frames=300,
            minimum_frames=32,
        )
        is None
    )
    assert (
        _minimum_frame_retry_fps(
            extracted_count=20,
            native_fps=30.0,
            total_native_frames=20,
            minimum_frames=32,
        )
        is None
    )


@pytest.mark.parametrize(
    ("extracted_count", "native_fps", "total_native_frames"),
    [
        (23, 29.88, 348),
        (59, 29.97, 879),
    ],
)
def test_minimum_frame_retry_fps_can_supply_256_frame_validation_videos(
    extracted_count, native_fps, total_native_frames
):
    retry = _minimum_frame_retry_fps(
        extracted_count=extracted_count,
        native_fps=native_fps,
        total_native_frames=total_native_frames,
        minimum_frames=256,
    )

    assert retry is not None
    assert 0 < retry <= native_fps
    duration = total_native_frames / native_fps
    assert retry * duration >= 258


def test_video_preprocessing_applies_to_main_and_isolated_roles():
    config = SimpleNamespace(
        **{
            role_name: SimpleNamespace(
                mm_processor_kwargs={"role": role_name}
            )
            for role_name in (
                "main_params",
                "planning_params",
                "general_params",
                "vlm_params",
                "vlm_grounding_params",
                "reflection_params",
            )
        }
    )

    _configure_video_role_preprocessing(config)

    for role_name in vars(config):
        assert getattr(config, role_name).mm_processor_kwargs == {
            "role": role_name,
            "max_num_tiles": 1,
            "video_as_images": True,
        }


def test_independent_main_and_auxiliary_reasoning_limits():
    role_names = (
        "main_params",
        "planning_params",
        "general_params",
        "vlm_params",
        "vlm_grounding_params",
        "reflection_params",
    )

    config = SimpleNamespace(
        **{
            role_name: SimpleNamespace(
                enable_thinking=None,
                thinking_token_budget=None,
                reasoning_budget=None,
            )
            for role_name in role_names
        }
    )

    _configure_reasoning_roles(
        config,
        main_enable_thinking=True,
        main_thinking_token_budget=512,
        main_reasoning_budget=512,
        auxiliary_enable_thinking=False,
        auxiliary_thinking_token_budget=None,
        auxiliary_reasoning_budget=None,
    )

    assert config.main_params.enable_thinking is True
    assert config.main_params.thinking_token_budget == 512
    assert config.main_params.reasoning_budget == 512
    for role_name in role_names[1:]:
        params = getattr(config, role_name)
        assert params.enable_thinking is False
        assert params.thinking_token_budget is None
        assert params.reasoning_budget is None


def test_finalizer_reasoning_limits_override_only_general_role():
    from responses_api_agents.spatialclaw_agent.app import (
        _configure_finalizer_role,
    )

    config = SimpleNamespace(
        main_params=SimpleNamespace(
            enable_thinking=False,
            thinking_token_budget=None,
            reasoning_budget=None,
        ),
        planning_params=SimpleNamespace(
            enable_thinking=False,
            thinking_token_budget=None,
            reasoning_budget=None,
        ),
        general_params=SimpleNamespace(
            enable_thinking=False,
            thinking_token_budget=None,
            reasoning_budget=None,
        ),
    )

    _configure_finalizer_role(
        config,
        enable_thinking=True,
        thinking_token_budget=None,
        reasoning_budget=2048,
    )

    assert config.general_params.enable_thinking is True
    assert config.general_params.thinking_token_budget is None
    assert config.general_params.reasoning_budget == 2048
    assert config.main_params.enable_thinking is False
    assert config.main_params.reasoning_budget is None
    assert config.planning_params.enable_thinking is False
    assert config.planning_params.reasoning_budget is None
