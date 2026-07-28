from types import SimpleNamespace

import pytest

from responses_api_agents.spatialclaw_agent.rl_adapter import (
    _ACTIVE_MAIN_SESSION,
    CapturedTurn,
    CaptureSession,
    _CompletionsProxy,
    _preserving_state_messages_to_openai,
    _replace_key_frames_with_videos,
    instrument_llm_client,
)


def _response(prompt_ids, generation_ids, logprobs, content="raw", reasoning="think"):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        prompt_token_ids=prompt_ids,
        generation_token_ids=generation_ids,
        generation_log_probs=logprobs,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def test_capture_keeps_exact_turn_tokens_and_only_new_prompt_media():
    first_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,first"},
    }
    second_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,second"},
    }
    first_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "q"}, first_image]},
    ]
    session = CaptureSession("session")
    session.capture(
        {
            "messages": first_messages,
            "extra_body": {"mm_processor_kwargs": {"max_num_patches": 512}},
        },
        _response([1, 2], [3], [-0.1]),
    )
    second_messages = first_messages + [
        {
            "role": "assistant",
            "content": "raw",
            "reasoning_content": "think",
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
        },
        {"role": "user", "content": [{"type": "text", "text": "feedback"}, second_image]},
    ]
    # SpatialClaw may rebuild an equivalent historical message with a different
    # representation. The exact token-prefix boundary remains authoritative and
    # prevents the initial image from being returned again as new turn media.
    second_messages[1] = {
        **second_messages[1],
        "rebuilt_by_graph": True,
    }
    session.capture(
        {
            "messages": second_messages,
            "extra_body": {
                "required_prefix_token_ids": [1, 2, 3],
                "required_prefix_message_count": 3,
            },
        },
        _response([1, 2, 3, 4], [5, 6], [-0.2, -0.3], content="next"),
    )

    assert session.turns[0].prompt_multimodal_content == [first_image]
    assert session.turns[0].prompt_mm_processor_kwargs == {"max_num_patches": 512}
    assert session.turns[1].prompt_multimodal_content == [second_image]
    assert session.turns[1].prompt_token_ids == [1, 2, 3, 4]
    assert session.turns[1].generation_token_ids == [5, 6]
    assert session.turns[1].generation_log_probs == [-0.2, -0.3]


def test_capture_rejects_token_logprob_mismatch():
    turn = CapturedTurn(
        content="x",
        reasoning_content=None,
        prompt_token_ids=[1],
        generation_token_ids=[2, 3],
        generation_log_probs=[-0.1],
    )
    with pytest.raises(RuntimeError, match="2 generation tokens but 1 logprobs"):
        turn.validate(0)


def test_video_aware_replaces_only_key_frame_media_and_preserves_references():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here are 2 key frames — a visual overview subset of InputImages.\n"
                        "Key frame mapping (context position → variable[index] → video frame):"
                    ),
                },
                {"type": "image_url", "image_url": {"url": "data:key-frame-1"}},
                {"type": "image_url", "image_url": {"url": "data:key-frame-2"}},
                {"type": "text", "text": "Reference images (1 total)."},
                {"type": "image_url", "image_url": {"url": "data:reference"}},
                {"type": "text", "text": "question"},
            ],
        }
    ]

    replaced, count = _replace_key_frames_with_videos(messages, ["/tmp/source.mp4"])

    assert count == 2
    content = replaced[0]["content"]
    assert content[1] == {
        "type": "video_url",
        "video_url": {"url": "file:///tmp/source.mp4"},
    }
    assert content[3] == {
        "type": "image_url",
        "image_url": {"url": "data:reference"},
    }
    assert "source video as model media" in content[0]["text"]
    assert "Tool key-frame mapping" in content[0]["text"]
    assert messages[0]["content"][1]["image_url"]["url"] == "data:key-frame-1"


@pytest.mark.asyncio
async def test_video_aware_main_call_captures_source_video_not_key_frames():
    responses = [
        _response([1, 2], [3], [-0.1]),
        _response([1, 2, 3, 4], [5], [-0.2], content="next"),
    ]

    class Completions:
        def __init__(self):
            self.calls = []

        async def create(self, *args, **kwargs):
            self.calls.append(kwargs)
            return responses[len(self.calls) - 1]

    completions = Completions()
    session = CaptureSession(
        "session",
        video_input_mode="video-aware",
        source_videos=["/tmp/source.mp4"],
    )
    token = _ACTIVE_MAIN_SESSION.set(session)
    try:
        first_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are 1 key frames — a visual overview subset of InputImages.",
                    },
                    {"type": "image_url", "image_url": {"url": "data:key-frame"}},
                    {"type": "text", "text": "question"},
                ],
            }
        ]
        await _CompletionsProxy(completions).create(
            messages=first_messages,
            extra_body={"mm_processor_kwargs": {"video_as_images": True}},
        )
        followup_image = {
            "type": "image_url",
            "image_url": {"url": "data:tool-observation"},
        }
        await _CompletionsProxy(completions).create(
            messages=first_messages
            + [
                {
                    "role": "assistant",
                    "content": "raw",
                    "reasoning_content": "think",
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "feedback"}, followup_image],
                },
            ],
            extra_body={
                "required_prefix_token_ids": [1, 2, 3],
                "mm_processor_kwargs": {"video_as_images": True},
            },
        )
    finally:
        _ACTIVE_MAIN_SESSION.reset(token)

    video_part = {
        "type": "video_url",
        "video_url": {"url": "file:///tmp/source.mp4"},
    }
    assert completions.calls[0]["messages"][0]["content"][1] == video_part
    assert completions.calls[1]["messages"][0]["content"][1] == video_part
    assert session.turns[0].prompt_multimodal_content == [video_part]
    assert session.turns[1].prompt_multimodal_content == [followup_image]
    assert completions.calls[1]["extra_body"]["mm_processor_kwargs"] == {
        "video_as_images": True,
        "video_as_images_frame_counts": [1],
    }


@pytest.mark.asyncio
async def test_keyframe_main_call_preserves_multiturn_video_frame_groups():
    class Completions:
        def __init__(self):
            self.kwargs = None

        async def create(self, *args, **kwargs):
            self.kwargs = kwargs
            return _response([1, 2], [3], [-0.1])

    def images(count, prefix):
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{prefix}-{index}"},
            }
            for index in range(count)
        ]

    completions = Completions()
    session = CaptureSession("session")
    token = _ACTIVE_MAIN_SESSION.set(session)
    try:
        await _CompletionsProxy(completions).create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question"},
                        *images(32, "keyframe"),
                    ],
                },
                {"role": "assistant", "content": "inspect"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "show result"},
                        *images(17, "show"),
                    ],
                },
                {"role": "assistant", "content": "inspect again"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "crop result"},
                        *images(1, "crop"),
                    ],
                },
            ],
            extra_body={"mm_processor_kwargs": {"video_as_images": True}},
        )
    finally:
        _ACTIVE_MAIN_SESSION.reset(token)

    assert completions.kwargs["extra_body"]["mm_processor_kwargs"] == {
        "video_as_images": True,
        "video_as_images_frame_counts": [32, 17, 1],
    }


def test_history_serializer_preserves_raw_reasoning_and_multimodal_content():
    messages = [
        SimpleNamespace(type="system", content="system"),
        SimpleNamespace(
            type="human",
            content=[{"type": "image_url", "image_url": {"url": "data:x"}}],
        ),
        SimpleNamespace(
            type="ai",
            content="verbatim",
            additional_kwargs={
                "reasoning_content": "hidden verbatim",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
                "generation_log_probs": [-0.1],
            },
        ),
    ]

    serialized = _preserving_state_messages_to_openai(messages)

    assert serialized[1]["content"] == messages[1].content
    assert serialized[2] == {
        "role": "assistant",
        "content": "verbatim",
        "reasoning_content": "hidden verbatim",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
        "generation_log_probs": [-0.1],
    }


def test_http_gym_endpoint_keeps_vllm_role_extensions():
    class Client:
        @staticmethod
        def _get_client(endpoint):
            return endpoint

        @staticmethod
        def _build_api_kwargs(params):
            # This is what SpatialClaw returns for a concrete HTTP base URL.
            return {"max_tokens": params.max_tokens}

    params = SimpleNamespace(
        max_tokens=128,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        skip_special_tokens=False,
        thinking_token_budget=64,
        enable_thinking=True,
        reasoning_budget=32,
        mm_processor_kwargs={"max_num_tiles": 1},
    )
    client = Client()

    instrument_llm_client(client)
    kwargs = client._build_api_kwargs(params)

    assert kwargs["extra_body"] == {
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
        "thinking_token_budget": 64,
        "mm_processor_kwargs": {"max_num_tiles": 1},
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_budget": 32,
            "truncate_history_thinking": False,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_budget": 32,
                "truncate_history_thinking": False,
            },
        },
    }

    disabled_params = SimpleNamespace(
        max_tokens=128,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        skip_special_tokens=None,
        thinking_token_budget=None,
        enable_thinking=False,
        reasoning_budget=None,
        mm_processor_kwargs=None,
    )
    disabled_kwargs = client._build_api_kwargs(disabled_params)
    assert disabled_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "truncate_history_thinking": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "truncate_history_thinking": False,
        },
    }


@pytest.mark.asyncio
async def test_auxiliary_session_strips_main_token_replay_metadata():
    class Completions:
        def __init__(self):
            self.kwargs = None

        async def create(self, *args, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(usage=None)

    completions = Completions()
    proxy = _CompletionsProxy(completions)
    await proxy.create(
        messages=[
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "main output",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
                "generation_log_probs": [-0.1],
            },
            {"role": "user", "content": "reflect independently"},
        ],
        extra_body={
            "required_prefix_token_ids": [1, 2, 3],
            "required_prefix_message_count": 2,
            "mm_processor_kwargs": {"video_as_images": True},
        },
    )

    assert completions.kwargs["messages"][1] == {
        "role": "assistant",
        "content": "main output",
    }
    assert completions.kwargs["extra_body"] == {"mm_processor_kwargs": {"video_as_images": True}}


@pytest.mark.asyncio
async def test_context_exceeded_main_call_is_not_captured_as_training():
    response = SimpleNamespace(
        context_length_exceeded=True,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    content=None,
                    prompt_token_ids=[1, 2, 3, 4],
                    generation_token_ids=[],
                    generation_log_probs=[],
                ),
            )
        ],
    )

    class Completions:
        async def create(self, *args, **kwargs):
            return response

    session = CaptureSession("session")
    token = _ACTIVE_MAIN_SESSION.set(session)
    try:
        returned = await _CompletionsProxy(Completions()).create(
            messages=[{"role": "user", "content": "too long"}],
            max_tokens=128,
        )
    finally:
        _ACTIVE_MAIN_SESSION.reset(token)

    assert returned is response
    assert session.turns == []
