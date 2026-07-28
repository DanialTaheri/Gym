# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RL compatibility hooks for an unmodified SpatialClaw checkout.

SpatialClaw normally normalizes successful assistant messages and may condense
failed messages.  That is useful for evaluation but invalid for NeMo RL: every
later prompt must begin with the exact tokens used by the preceding policy
calls.  This module installs process-local, version-checked hooks before the
workflow graph is compiled and records only calls made by the main agent node.
"""

from __future__ import annotations

import contextvars
import copy
import importlib
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and name in extra:
        return extra[name]
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", part)))
        return "".join(parts)
    return str(content)


def _media_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {
                "image",
                "image_url",
                "input_image",
                "input_video",
                "video",
                "video_url",
            }:
                media.append(copy.deepcopy(part))
    return media


def _video_as_images_frame_counts(messages: list[dict[str, Any]]) -> list[int]:
    """Return image counts at the message boundaries seen by the main policy.

    vLLM otherwise flattens every ``image_url`` in a multi-turn request into
    one synthetic video.  SpatialClaw introduces visual observations between
    sampled assistant turns, so that flattening changes the Conv3D grouping of
    historical frames on later requests.  Carry the original message groups to
    the model server so every turn and policy replay use the same temporal
    boundaries.
    """
    frame_counts: list[int] = []
    for raw_message in messages:
        message = (
            raw_message.model_dump(exclude_none=True)
            if hasattr(raw_message, "model_dump")
            else raw_message
        )
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count = sum(
            1
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"image", "image_url", "input_image"}
        )
        if count:
            frame_counts.append(count)
    return frame_counts


def _video_url(value: str) -> str:
    """Return an OpenAI-compatible URL for a materialized source video."""
    if "://" in value or value.startswith("data:"):
        return value
    return Path(value).expanduser().resolve().as_uri()


def _replace_key_frames_with_videos(
    messages: list[dict[str, Any]], source_videos: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """Replace SpatialClaw's initial key-frame media with source videos.

    SpatialClaw keeps all extracted frames in ``InputImages`` for Python tools,
    while its first model message contains an evenly selected visual overview.
    Video-aware mode changes only that model-facing overview. Reference images
    that follow the key-frame block remain independent image media.
    """
    replaced_messages = copy.deepcopy(messages)
    for message in replaced_messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        content = message["content"]
        for header_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
                continue
            header = str(part.get("text", ""))
            match = re.search(r"Here are (\d+) key frames", header)
            if match is None:
                continue
            key_frame_count = int(match.group(1))
            cursor = header_index + 1
            removed = 0
            while cursor < len(content) and removed < key_frame_count:
                media_part = content[cursor]
                if not isinstance(media_part, dict) or media_part.get("type") not in {
                    "image",
                    "image_url",
                    "input_image",
                }:
                    break
                del content[cursor]
                removed += 1
            if removed != key_frame_count:
                raise RuntimeError(
                    "SpatialClaw video-aware media replacement found "
                    f"{removed} key-frame images after a {key_frame_count}-frame header"
                )
            part["text"] = (
                re.sub(
                    r"Here are \d+ key frames — a visual overview subset of",
                    "Here is the source video as model media. SpatialClaw's tool "
                    "kernel separately exposes the selected key-frame overview of",
                    header,
                    count=1,
                )
                .replace(
                    "Key frame mapping (context position → variable[index] → video frame):",
                    "Tool key-frame mapping (selection position → variable[index] → video frame):",
                )
            )
            video_parts = [
                {"type": "video_url", "video_url": {"url": _video_url(value)}}
                for value in source_videos
            ]
            content[cursor:cursor] = video_parts
            return replaced_messages, removed
    raise RuntimeError(
        "SpatialClaw video-aware mode could not find the initial key-frame media block"
    )


@dataclass
class CapturedTurn:
    content: str
    reasoning_content: str | None
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]
    prompt_multimodal_content: list[dict[str, Any]] = field(default_factory=list)
    prompt_mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    request_required_prefix_length: int = 0
    request_required_prefix_message_count: int = 0
    request_message_prefix_length: int = 0
    finish_reason: str | None = None

    def validate(self, turn_index: int) -> None:
        if not self.prompt_token_ids:
            raise RuntimeError(
                f"SpatialClaw main turn {turn_index} has no prompt_token_ids; "
                "the Gym vLLM model must enable return_token_id_information"
            )
        if not self.generation_token_ids:
            raise RuntimeError(f"SpatialClaw main turn {turn_index} has no generation_token_ids")
        if len(self.generation_token_ids) != len(self.generation_log_probs):
            raise RuntimeError(
                "SpatialClaw main turn "
                f"{turn_index} has {len(self.generation_token_ids)} generation "
                f"tokens but {len(self.generation_log_probs)} logprobs"
            )


@dataclass
class CaptureSession:
    session_id: str
    video_input_mode: str = "key-frame-aware"
    source_videos: list[str] = field(default_factory=list)
    turns: list[CapturedTurn] = field(default_factory=list)
    previous_prompt_messages: list[dict[str, Any]] = field(default_factory=list)

    def capture(self, request_kwargs: dict[str, Any], response: Any) -> None:
        messages = copy.deepcopy(request_kwargs.get("messages") or [])
        extra_body = request_kwargs.get("extra_body") or {}
        required_prefix_message_count = int(
            extra_body.get("required_prefix_message_count") or 0
        )
        if required_prefix_message_count:
            if required_prefix_message_count > len(messages):
                raise RuntimeError(
                    "SpatialClaw required_prefix_message_count exceeds the request "
                    f"message count: {required_prefix_message_count} > {len(messages)}"
                )
            # This boundary accompanies required_prefix_token_ids and is the
            # authoritative end of the exact on-policy history. SpatialClaw can
            # rebuild equivalent historical message objects between graph turns,
            # so object equality is not reliable for deciding which media are new.
            prefix_len = required_prefix_message_count
        else:
            prefix_len = 0
            while (
                prefix_len < len(self.previous_prompt_messages)
                and prefix_len < len(messages)
                and self.previous_prompt_messages[prefix_len] == messages[prefix_len]
            ):
                prefix_len += 1
        new_prompt_messages = messages[prefix_len:]
        self.previous_prompt_messages = messages

        choices = _field(response, "choices", []) or []
        if not choices:
            raise RuntimeError("SpatialClaw policy response did not contain a choice")
        choice = choices[0]
        message = _field(choice, "message")
        if message is None:
            raise RuntimeError("SpatialClaw policy response did not contain a message")

        prompt_token_ids = _field(message, "prompt_token_ids", []) or []
        if not prompt_token_ids:
            prompt_token_ids = _field(response, "prompt_token_ids", []) or []
        generation_token_ids = _field(message, "generation_token_ids", []) or []
        if not generation_token_ids:
            generation_token_ids = _field(choice, "token_ids", []) or []
        generation_log_probs = _field(message, "generation_log_probs", []) or []
        if not generation_log_probs:
            logprobs = _field(choice, "logprobs")
            logprob_content = _field(logprobs, "content", []) or []
            generation_log_probs = [float(_field(item, "logprob", 0.0)) for item in logprob_content]

        mm_processor_kwargs = extra_body.get("mm_processor_kwargs") or {}
        message_prefix_length = 0
        for request_message in reversed(messages):
            message_prompt_ids = request_message.get("prompt_token_ids") or []
            if message_prompt_ids:
                message_prefix_length = len(message_prompt_ids) + len(
                    request_message.get("generation_token_ids") or []
                )
                break
        reasoning = _field(message, "reasoning_content")
        if reasoning is None:
            reasoning = _field(message, "reasoning")
        self.turns.append(
            CapturedTurn(
                content=_content_text(_field(message, "content", "")),
                reasoning_content=_content_text(reasoning) or None,
                prompt_token_ids=[int(value) for value in prompt_token_ids],
                generation_token_ids=[int(value) for value in generation_token_ids],
                generation_log_probs=[float(value) for value in generation_log_probs],
                prompt_multimodal_content=_media_from_messages(new_prompt_messages),
                prompt_mm_processor_kwargs=copy.deepcopy(mm_processor_kwargs),
                request_required_prefix_length=len(extra_body.get("required_prefix_token_ids") or []),
                request_required_prefix_message_count=int(extra_body.get("required_prefix_message_count") or 0),
                request_message_prefix_length=message_prefix_length,
                finish_reason=_field(choice, "finish_reason"),
            )
        )


_ACTIVE_MAIN_SESSION: contextvars.ContextVar[CaptureSession | None] = contextvars.ContextVar(
    "spatialclaw_active_main_session", default=None
)
_SESSIONS: dict[str, CaptureSession] = {}
_HOOKS_INSTALLED = False
_ORIGINAL_LLM_STEP_NODE = None


class _CompletionsProxy:
    def __init__(self, completions: Any):
        self._completions = completions

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        session = _ACTIVE_MAIN_SESSION.get()
        call_kind = "main" if session is not None else "auxiliary"
        messages = kwargs.get("messages") or []

        if session is not None:
            if session.video_input_mode == "video-aware":
                kwargs = dict(kwargs)
                messages, replaced_count = _replace_key_frames_with_videos(
                    messages, session.source_videos
                )
                kwargs["messages"] = messages
                print(
                    "[spatialclaw_llm] kind=main input_mode=video-aware "
                    f"replaced_key_frames={replaced_count} "
                    f"source_videos={len(session.source_videos)}",
                    file=sys.stderr,
                    flush=True,
                )
            # Determine this boundary before Gym/OpenAI Pydantic schemas
            # normalize away the training-only token metadata on historical
            # assistant messages. The exact prefix attached by
            # ``_rl_llm_step_node`` ends at the most recent message carrying
            # sampled token IDs. This correction belongs only to trainable
            # main-agent calls.
            required_prefix_message_count = 0
            for index in reversed(range(len(messages))):
                message = messages[index]
                if hasattr(message, "model_dump"):
                    message = message.model_dump(exclude_none=True)
                if isinstance(message, dict) and message.get("prompt_token_ids"):
                    required_prefix_message_count = index + 1
                    break
            if required_prefix_message_count:
                kwargs = dict(kwargs)
                extra_body = copy.deepcopy(kwargs.get("extra_body") or {})
                extra_body["required_prefix_message_count"] = required_prefix_message_count
                kwargs["extra_body"] = extra_body

            extra_body = copy.deepcopy(kwargs.get("extra_body") or {})
            mm_processor_kwargs = copy.deepcopy(
                extra_body.get("mm_processor_kwargs") or {}
            )
            if mm_processor_kwargs.get("video_as_images"):
                frame_counts = _video_as_images_frame_counts(messages)
                if frame_counts:
                    kwargs = dict(kwargs)
                    mm_processor_kwargs["video_as_images_frame_counts"] = (
                        frame_counts
                    )
                    extra_body["mm_processor_kwargs"] = mm_processor_kwargs
                    kwargs["extra_body"] = extra_body
        else:
            # Planner, reflection, force-termination VLM, and vlm.* requests
            # are independent model sessions. They may consume the textual
            # conversation, but must never inherit main-session token IDs or
            # exact-prefix replay controls.
            kwargs = dict(kwargs)
            isolated_messages = copy.deepcopy(messages)
            for message in isolated_messages:
                if not isinstance(message, dict):
                    continue
                for field_name in (
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                ):
                    message.pop(field_name, None)
            kwargs["messages"] = isolated_messages
            messages = isolated_messages
            extra_body = copy.deepcopy(kwargs.get("extra_body") or {})
            extra_body.pop("required_prefix_token_ids", None)
            extra_body.pop("required_prefix_message_count", None)
            kwargs["extra_body"] = extra_body

        started = perf_counter()
        try:
            response = await self._completions.create(*args, **kwargs)
        except Exception:
            print(
                f"[spatialclaw_llm] kind={call_kind} status=failed "
                f"max_tokens={kwargs.get('max_tokens')} "
                f"messages={len(messages)} elapsed_sec={perf_counter() - started:.3f}",
                file=sys.stderr,
                flush=True,
            )
            raise
        usage = _field(response, "usage")
        print(
            f"[spatialclaw_llm] kind={call_kind} status=completed "
            f"max_tokens={kwargs.get('max_tokens')} "
            f"messages={len(messages)} output_tokens="
            f"{_field(usage, 'completion_tokens', _field(usage, 'output_tokens', 0))} "
            f"elapsed_sec={perf_counter() - started:.3f}",
            file=sys.stderr,
            flush=True,
        )
        if session is not None:
            if _field(response, "context_length_exceeded", False):
                # No model generation occurred.  Do not turn an over-length
                # attempted prompt into a trainable turn: _rl_llm_step_node
                # will retain it as environment feedback while the last exact
                # sampled prefix remains untouched.
                print(
                    "[spatialclaw_llm] kind=main capture=skipped_context_length_exceeded",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                session.capture(kwargs, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _ChatProxy:
    def __init__(self, chat: Any):
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _OpenAIProxy:
    def __init__(self, client: Any):
        self._client = client
        self.chat = _ChatProxy(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _preserving_state_messages_to_openai(messages, agent_config=None, state=None) -> list[dict[str, Any]]:
    """Serialize the complete immutable history, including separated reasoning."""
    result: list[dict[str, Any]] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        role = {"system": "system", "human": "user", "ai": "assistant"}.get(message_type, "user")
        content = getattr(message, "content", "")
        item: dict[str, Any] = {
            "role": role,
            "content": content if isinstance(content, (str, list)) else str(content),
        }
        if role == "assistant":
            additional = getattr(message, "additional_kwargs", None) or {}
            reasoning = additional.get("reasoning_content")
            if reasoning is not None:
                item["reasoning_content"] = reasoning
            # NeMo RL's vLLM chat server uses these fields to replace the
            # re-tokenized assistant prefix with the exact tokens sampled on
            # the preceding call. This is the same on-policy correction used
            # by the Hermes/SWE-style Gym harnesses.
            for field_name in (
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
            ):
                if field_name in additional:
                    item[field_name] = copy.deepcopy(additional[field_name])
        result.append(item)
    return result


async def _rl_llm_step_node(state, config):
    """Run SpatialClaw's node, then restore the exact model message in state."""
    from langchain_core.messages import AIMessage, HumanMessage

    session_id = str(state.get("session_id") or "")
    session = _SESSIONS.get(session_id)
    if session is None:
        raise RuntimeError(f"No SpatialClaw RL capture session registered for {session_id!r}")

    before = len(session.turns)
    llm_client = config.get("configurable", {}).get("llm_client")
    previous_required_prefix = getattr(llm_client, "_nemo_gym_required_prefix_token_ids", None)
    required_prefix = None
    if session.turns:
        previous_turn = session.turns[-1]
        required_prefix = copy.deepcopy(previous_turn.prompt_token_ids) + copy.deepcopy(
            previous_turn.generation_token_ids
        )
    if llm_client is not None:
        llm_client._nemo_gym_required_prefix_token_ids = required_prefix
    token = _ACTIVE_MAIN_SESSION.set(session)
    try:
        result = await _ORIGINAL_LLM_STEP_NODE(state, config)
    finally:
        _ACTIVE_MAIN_SESSION.reset(token)
        if llm_client is not None:
            llm_client._nemo_gym_required_prefix_token_ids = previous_required_prefix

    messages = list(result.get("messages") or [])
    if len(session.turns) == before:
        # An infrastructure failure produced no policy tokens. Keep the error as
        # environment context, never as a fabricated assistant generation.
        result["messages"] = [
            HumanMessage(content=f"[Policy call failed] {getattr(message, 'content', message)}")
            if isinstance(message, AIMessage)
            else message
            for message in messages
        ]
        return result

    turn = session.turns[-1]
    raw_message = AIMessage(
        content=turn.content,
        additional_kwargs={
            "reasoning_content": turn.reasoning_content,
            "prompt_token_ids": turn.prompt_token_ids,
            "generation_token_ids": turn.generation_token_ids,
            "generation_log_probs": turn.generation_log_probs,
        },
    )
    replaced = False
    preserved_messages = []
    for message in messages:
        if isinstance(message, AIMessage) and not replaced:
            preserved_messages.append(raw_message)
            replaced = True
        else:
            preserved_messages.append(message)
    if not replaced:
        preserved_messages.insert(0, raw_message)
    result["messages"] = preserved_messages
    return result


def install_spatialclaw_rl_hooks() -> None:
    """Install the immutable-history hooks once in the agent-server process."""
    global _HOOKS_INSTALLED, _ORIGINAL_LLM_STEP_NODE
    if _HOOKS_INSTALLED:
        return

    # ``spatial_agent.nodes.__init__`` exports a function named
    # ``llm_step_node``. A dotted ``import ... as`` can therefore bind that
    # package attribute instead of the submodule. Resolve both modules
    # explicitly so the hook patches the globals used by the workflow.
    llm_step_module = importlib.import_module("spatial_agent.nodes.llm_step_node")
    workflow_module = importlib.import_module("spatial_agent.workflow")

    original = getattr(llm_step_module, "llm_step_node", None)
    serializer = getattr(llm_step_module, "_state_messages_to_openai", None)
    if original is None or serializer is None or not hasattr(workflow_module.SpatialAgentWorkflow, "_build_graph"):
        raise RuntimeError(
            "The mounted SpatialClaw checkout is incompatible with the Gym RL adapter: "
            "expected llm_step_node, _state_messages_to_openai, and SpatialAgentWorkflow._build_graph"
        )

    _ORIGINAL_LLM_STEP_NODE = original
    llm_step_module._state_messages_to_openai = _preserving_state_messages_to_openai
    workflow_module.llm_step_node = _rl_llm_step_node
    _HOOKS_INSTALLED = True


def instrument_llm_client(client: Any) -> None:
    """Wrap SpatialClaw's OpenAI transport and force reasoning preservation."""
    if getattr(client, "_nemo_gym_rl_instrumented", False):
        return
    if not hasattr(client, "_get_client") or not hasattr(client, "_build_api_kwargs"):
        raise RuntimeError("Mounted SpatialClaw LLMClient lacks required adapter hooks")

    original_get_client = client._get_client
    original_build_api_kwargs = client._build_api_kwargs
    proxies: dict[str, _OpenAIProxy] = {}

    def patched_get_client(_self, endpoint: str):
        if endpoint not in proxies:
            proxies[endpoint] = _OpenAIProxy(original_get_client(endpoint))
        return proxies[endpoint]

    def patched_build_api_kwargs(_self, params):
        kwargs = original_build_api_kwargs(params)
        # SpatialClaw only emits vLLM extensions when llm_base_url is the
        # literal string "vllm". Gym supplies a concrete HTTP proxy URL, so
        # restore the role parameters that would otherwise be silently lost.
        extra_body = kwargs.setdefault("extra_body", {})
        for name in (
            "top_k",
            "min_p",
            "repetition_penalty",
            "skip_special_tokens",
            "thinking_token_budget",
            "mm_processor_kwargs",
        ):
            value = getattr(params, name, None)
            if value is not None:
                extra_body[name] = copy.deepcopy(value)

        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        enable_thinking = getattr(params, "enable_thinking", None)
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        reasoning_budget = getattr(params, "reasoning_budget", None)
        if reasoning_budget is not None:
            chat_template_kwargs["reasoning_budget"] = reasoning_budget
        chat_template_kwargs["truncate_history_thinking"] = False
        # vLLM expands request ``chat_template_kwargs`` into top-level Jinja
        # variables.  The external NanoV3 template used by the 16K SFT stage
        # instead follows Megatron's processor convention and reads the
        # mapping from a Jinja variable named ``chat_template_kwargs``.  Send
        # both representations so role controls work with either template
        # convention.  Without the nested copy, auxiliary ``enable_thinking
        # = false`` still renders an opening <think> and force-termination can
        # consume its entire answer budget as private reasoning.
        chat_template_kwargs["chat_template_kwargs"] = copy.deepcopy(
            chat_template_kwargs
        )

        # NeMo RL replaces the re-tokenized history up to the most recent
        # assistant message with this exact on-policy prefix.  Send it
        # explicitly as a vLLM request extension as well as embedding the
        # token metadata in the assistant message.  The explicit field is
        # robust to OpenAI-compatible proxy schemas that select the ordinary
        # assistant-message union variant and discard training-only keys.
        required_prefix = getattr(_self, "_nemo_gym_required_prefix_token_ids", None)
        if required_prefix:
            extra_body["required_prefix_token_ids"] = copy.deepcopy(required_prefix)
        return kwargs

    client._get_client = types.MethodType(patched_get_client, client)
    client._build_api_kwargs = types.MethodType(patched_build_api_kwargs, client)
    client._nemo_gym_rl_instrumented = True


def start_capture(
    session_id: str,
    *,
    video_input_mode: str = "key-frame-aware",
    source_videos: list[str] | None = None,
) -> CaptureSession:
    if session_id in _SESSIONS:
        raise RuntimeError(f"Duplicate SpatialClaw RL session id: {session_id}")
    source_videos = list(source_videos or [])
    if video_input_mode == "video-aware" and not source_videos:
        raise ValueError("SpatialClaw video-aware mode requires at least one source video")
    session = CaptureSession(
        session_id=session_id,
        video_input_mode=video_input_mode,
        source_videos=source_videos,
    )
    _SESSIONS[session_id] = session
    return session


def finish_capture(session_id: str) -> list[CapturedTurn]:
    session = _SESSIONS.pop(session_id, None)
    if session is None:
        raise RuntimeError(f"Unknown SpatialClaw RL session id: {session_id}")
    if not session.turns:
        raise RuntimeError("SpatialClaw completed without a trainable main-agent turn")
    for index, turn in enumerate(session.turns):
        turn.validate(index)
    return session.turns


def discard_capture(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
