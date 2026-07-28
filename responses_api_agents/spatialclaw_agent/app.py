# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym RL harness for the SpatialClaw image/video agent."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import urllib.request
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.spatialclaw_agent.rl_adapter import (
    discard_capture,
    finish_capture,
    install_spatialclaw_rl_hooks,
    instrument_llm_client,
    start_capture,
)


class SpatialClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    spatialclaw_root: str = Field(default_factory=lambda: os.environ.get("SPATIALCLAW_ROOT", ""))
    model_name: str = ""
    dataset_config: str | None = None
    model_config_path: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    main_mm_processor_kwargs: dict[str, Any] = Field(default_factory=dict)
    video_input_mode: Literal["key-frame-aware", "video-aware"] = "key-frame-aware"
    compact_main_prompt: bool = False
    compact_planner_prompt: bool = False
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_output_tokens: int | None = None
    auxiliary_max_output_tokens: int | None = None
    main_enable_thinking: bool | None = None
    main_thinking_token_budget: int | None = None
    main_reasoning_budget: int | None = None
    auxiliary_enable_thinking: bool | None = None
    auxiliary_thinking_token_budget: int | None = None
    auxiliary_reasoning_budget: int | None = None
    finalizer_max_output_tokens: int | None = None
    finalizer_enable_thinking: bool | None = None
    finalizer_thinking_token_budget: int | None = None
    finalizer_reasoning_budget: int | None = None
    concurrency: int = 4
    timeout: int = 1800
    workspace_root: str = "outputs/spatialclaw_agent"
    keep_workspaces: bool = False


class SpatialClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SpatialClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    termination_reason: str | None = None


def _dump_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported Responses input item: {type(item)!r}")


def _configure_video_role_preprocessing(spatial_config: Any) -> None:
    """Use the 16K SFT Conv3D path for every video-origin model session."""
    for role_name in (
        "main_params",
        "planning_params",
        "general_params",
        "vlm_params",
        "vlm_grounding_params",
        "reflection_params",
    ):
        role_params = getattr(spatial_config, role_name)
        mm_processor_kwargs = dict(
            getattr(role_params, "mm_processor_kwargs", {}) or {}
        )
        mm_processor_kwargs["max_num_tiles"] = 1
        mm_processor_kwargs["video_as_images"] = True
        role_params.mm_processor_kwargs = mm_processor_kwargs


def _configure_compact_planner_prompt(spatial_config: Any) -> None:
    """Fit the isolated text-only planner in a 16K model context.

    SpatialClaw's planning prompt has both a concise planning strategy and a
    second, full reference catalog for every available tool.  The catalog is
    useful with long-context models but makes the otherwise text-only planner
    exceed a 16K window before generation starts.  Use SpatialClaw's native
    prompt-section control to omit only that duplicate catalog.  The planner's
    question, frame metadata, tool-selection strategy, API protocols, rules,
    and checklist remain unchanged.
    """
    ablations = copy.deepcopy(
        getattr(spatial_config, "prompt_section_ablations", {}) or {}
    )
    excluded = list(ablations.get("exclude", []) or [])
    if "planning_available_tools" not in excluded:
        excluded.append("planning_available_tools")
    ablations["exclude"] = excluded
    spatial_config.prompt_section_ablations = ablations


def _configure_compact_main_prompt(spatial_config: Any) -> None:
    """Keep the trainable main conversation usable in a 16K window.

    SpatialClaw's main prompt contains the complete API reference for every
    CPU/GPU tool in addition to the dedicated ``show`` and ``vlm`` sections.
    The isolated planner already selects the tools for a trajectory, so that
    catalog is duplicate context for the main policy.  Omitting only the
    catalog leaves the response contract, code rules, ReturnAnswer protocol,
    workflow, media metadata, show API, and VLM API intact.  Most importantly,
    it reserves context for exact, immutable assistant/tool history instead of
    forcing later policy calls to overflow the model window.
    """
    ablations = copy.deepcopy(
        getattr(spatial_config, "prompt_section_ablations", {}) or {}
    )
    excluded = list(ablations.get("exclude", []) or [])
    if "available_tools" not in excluded:
        excluded.append("available_tools")
    ablations["exclude"] = excluded
    spatial_config.prompt_section_ablations = ablations


def _configure_reasoning_roles(
    spatial_config: Any,
    *,
    main_enable_thinking: bool | None,
    main_thinking_token_budget: int | None,
    main_reasoning_budget: int | None,
    auxiliary_enable_thinking: bool | None,
    auxiliary_thinking_token_budget: int | None,
    auxiliary_reasoning_budget: int | None,
) -> None:
    """Apply independent main and auxiliary reasoning controls."""
    auxiliary_role_params = (
        spatial_config.planning_params,
        spatial_config.general_params,
        spatial_config.vlm_params,
        spatial_config.vlm_grounding_params,
        spatial_config.reflection_params,
    )
    main_settings = {
        "enable_thinking": main_enable_thinking,
        "thinking_token_budget": main_thinking_token_budget,
        "reasoning_budget": main_reasoning_budget,
    }
    auxiliary_settings = {
        "enable_thinking": auxiliary_enable_thinking,
        "thinking_token_budget": auxiliary_thinking_token_budget,
        "reasoning_budget": auxiliary_reasoning_budget,
    }
    for name, value in main_settings.items():
        if value is not None:
            setattr(spatial_config.main_params, name, value)
    for params in auxiliary_role_params:
        for name, value in auxiliary_settings.items():
            if value is not None:
                setattr(params, name, value)


def _configure_finalizer_role(
    spatial_config: Any,
    *,
    enable_thinking: bool | None,
    thinking_token_budget: int | None,
    reasoning_budget: int | None,
) -> None:
    """Override only SpatialClaw's isolated force-termination VLM session."""
    settings = {
        "enable_thinking": enable_thinking,
        "thinking_token_budget": thinking_token_budget,
        "reasoning_budget": reasoning_budget,
    }
    for name, value in settings.items():
        if value is not None:
            setattr(spatial_config.general_params, name, value)


def _part_url(part: dict[str, Any]) -> str | None:
    for key in ("image_url", "video_url", "image", "video", "url"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for nested_key in ("url", "file_url", "path"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested
    return None


def _suffix_for_url(url: str, default: str) -> str:
    if url.startswith("data:"):
        media_type = url[5:].split(";", 1)[0]
        subtype = media_type.split("/", 1)[-1].split("+", 1)[0]
        return f".{subtype}" if subtype else default
    suffix = Path(urlparse(url).path).suffix
    return suffix or default


def _materialize_url(url: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file://"):
        return unquote(urlparse(url).path)
    if url.startswith("data:"):
        _, payload = url.split(",", 1)
        output_path.write_bytes(base64.b64decode(payload))
        return str(output_path)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=120) as response:
            output_path.write_bytes(response.read())
        return str(output_path)
    return str(Path(url).expanduser().resolve())


def _metadata(body: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    metadata = body.metadata or {}
    raw = metadata.get("spatialclaw") if isinstance(metadata, dict) else None
    if raw is None:
        return {}
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError("responses_create_params.metadata.spatialclaw must encode a JSON object")
    return parsed


def _session_id(value: Any) -> str:
    session_id = str(value or uuid4().hex)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
        raise ValueError(
            "SpatialClaw metadata.session_id must be 1-128 filename-safe "
            "characters (letters, digits, dot, underscore, or hyphen)"
        )
    return session_id


def _finish_timed_out_capture(session_id: str) -> tuple[list[Any], dict[str, Any]]:
    """Finish a real partial trajectory without inventing a terminal answer."""
    turns = finish_capture(session_id)
    return turns, {
        "final_answer": {"text": ""},
        "termination_reason": "timeout",
    }


def _minimum_frame_retry_fps(
    *,
    extracted_count: int,
    native_fps: float,
    total_native_frames: int,
    minimum_frames: int,
) -> float | None:
    """Return an adaptive extraction cap when the first pass is too sparse."""
    if extracted_count >= minimum_frames or minimum_frames <= 0:
        return None
    if native_fps <= 0 or total_native_frames < minimum_frames:
        return None
    duration = total_native_frames / native_fps
    if duration <= 0:
        return None
    # Ask ffmpeg for two extra frames because its fps filter rounds boundaries
    # differently across containers/codecs. Never request above native FPS.
    return min(native_fps, (minimum_frames + 2) / duration)


def _extract_request_input(body: NeMoGymResponseCreateParamsNonStreaming) -> tuple[str, list[str], list[str]]:
    items = [NeMoGymEasyInputMessage(role="user", content=body.input)] if isinstance(body.input, str) else body.input
    text_parts: list[str] = []
    image_urls: list[str] = []
    video_urls: list[str] = []
    for item in items:
        message = _dump_item(item)
        if message.get("role") not in {"user", "developer", "system"}:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                text_parts.append(str(part.get("text", "")))
            elif part_type in {"image", "image_url", "input_image"}:
                url = _part_url(part)
                if url:
                    image_urls.append(url)
            elif part_type in {"video", "video_url", "input_video"}:
                url = _part_url(part)
                if url:
                    video_urls.append(url)
    return "\n\n".join(part for part in text_parts if part), image_urls, video_urls


def _write_capture_manifest(
    session_dir: Path,
    turns: list[Any],
    result: dict[str, Any] | None = None,
    video_input_mode: str = "key-frame-aware",
) -> None:
    """Persist compact evidence for token-history and media-turn verification."""
    manifest_turns: list[dict[str, Any]] = []
    previous_sequence: list[int] = []
    for index, turn in enumerate(turns):
        media = []
        for part in turn.prompt_multimodal_content:
            url = _part_url(part) or ""
            media.append(
                {
                    "type": str(part.get("type") or ""),
                    "url_scheme": url.split(":", 1)[0] if ":" in url else "path",
                    "url_chars": len(url),
                    "url_sha256": hashlib.sha256(url.encode()).hexdigest() if url else "",
                }
            )
        prompt_ids = [int(value) for value in turn.prompt_token_ids]
        generation_ids = [int(value) for value in turn.generation_token_ids]
        manifest_turns.append(
            {
                "index": index,
                "content": turn.content,
                "reasoning_content": turn.reasoning_content,
                "prompt_token_ids": prompt_ids,
                "generation_token_ids": generation_ids,
                "generation_log_probs": [float(value) for value in turn.generation_log_probs],
                "finish_reason": turn.finish_reason,
                "prompt_mm_processor_kwargs": turn.prompt_mm_processor_kwargs,
                "prompt_media": media,
                "request_required_prefix_length": turn.request_required_prefix_length,
                "request_required_prefix_message_count": (
                    turn.request_required_prefix_message_count
                ),
                "request_message_prefix_length": turn.request_message_prefix_length,
                "history_prefix_length": len(previous_sequence),
                "history_prefix_matches": prompt_ids[: len(previous_sequence)] == previous_sequence,
            }
        )
        previous_sequence = prompt_ids + generation_ids
    result = result or {}
    final_answer = result.get("final_answer") or {}
    (session_dir / "rl_capture.json").write_text(
        json.dumps(
            {
                "video_input_mode": video_input_mode,
                "turn_count": len(turns),
                "turns": manifest_turns,
                "final_answer": str(final_answer.get("text") or ""),
                "termination_reason": str(result.get("termination_reason") or ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class SpatialClawAgent(SimpleResponsesAPIAgent):
    config: SpatialClawAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resolve_spatialclaw_root(self) -> Path:
        root_value = self.config.spatialclaw_root or os.environ.get("SPATIALCLAW_ROOT", "")
        if not root_value:
            raise RuntimeError("SPATIALCLAW_ROOT or config.spatialclaw_root is required")
        root = Path(root_value).expanduser().resolve()
        if not (root / "spatial_agent" / "workflow.py").is_file():
            raise RuntimeError(f"Invalid SpatialClaw checkout: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # SpatialClaw executes tool code in a child Jupyter kernel. Updating
        # only this server process's ``sys.path`` is insufficient because the
        # kernel deserializes ``spatial_agent`` objects in a fresh interpreter.
        # Export the checkout before the workflow creates its kernel pool so
        # every child process inherits the same import path.
        pythonpath = [
            part
            for part in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if part
        ]
        if str(root) not in pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join([str(root), *pythonpath])
        return root

    def _model_base_url(self) -> str:
        server = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.model_server.name,
        )
        return f"http://{server.host}:{server.port}/v1"

    @staticmethod
    def _config_path(root: Path, value: str | None, kind: str) -> str | None:
        if not value:
            return None
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / "spatial_agent" / "config" / kind / value
        if candidate.suffix != ".json":
            candidate = candidate.with_suffix(".json")
        if not candidate.is_file():
            raise FileNotFoundError(f"SpatialClaw {kind} config not found: {candidate}")
        return str(candidate)

    def _build_spatialclaw_config(
        self,
        root: Path,
        run_metadata: dict[str, Any],
        session_dir: Path,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ):
        from spatial_agent.config import SpatialAgentConfig

        config = SpatialAgentConfig()
        dataset_config = run_metadata.get("dataset_config") or self.config.dataset_config
        model_config = run_metadata.get("model_config") or self.config.model_config_path
        dataset_path = self._config_path(root, dataset_config, "dataset")
        model_path = self._config_path(root, model_config, "model")
        if dataset_path:
            config.update_from_dataset_json(dataset_path)
        if model_path:
            config.update_from_model_json(model_path)

        overrides = dict(self.config.config_overrides)
        overrides.update(run_metadata.get("config_overrides") or {})
        for name, value in overrides.items():
            if not hasattr(config, name):
                raise ValueError(f"Unknown SpatialClaw config override: {name}")
            setattr(config, name, value)

        config.llm_base_url = self._model_base_url()
        config.llm_model = self.config.model_name or self.config.model_server.name
        config.llm_api_key = "gym"  # pragma: allowlist secret
        config.work_dir = str(session_dir)
        config.concurrency = 1
        config.generate_report = False
        config.enable_logging = bool(run_metadata.get("enable_logging", False))

        # A fixed image budget makes vLLM and the policy-side HF processor use
        # identical deterministic tiling for initial keyframes and later
        # show() images. The exact kwargs are also captured with every turn.
        main_mm_processor_kwargs = copy.deepcopy(self.config.main_mm_processor_kwargs)
        main_mm_processor_kwargs.update(
            run_metadata.get("main_mm_processor_kwargs") or {}
        )
        if main_mm_processor_kwargs:
            config.main_params.mm_processor_kwargs = main_mm_processor_kwargs

        if self.config.compact_main_prompt:
            _configure_compact_main_prompt(config)
        if self.config.compact_planner_prompt:
            _configure_compact_planner_prompt(config)

        # NeMo RL's generation worker rejects per-request sampling that differs
        # from the rollout config, including requests from isolated planner,
        # reflection, and VLM sessions. Keep those conversations isolated while
        # applying the same on-policy temperature/top-p to every role.
        temperature = (
            body.temperature
            if body.temperature is not None
            else self.config.temperature
        )
        top_p = body.top_p if body.top_p is not None else self.config.top_p
        requested_max_output_tokens = body.max_output_tokens
        configured_max_output_tokens = self.config.max_output_tokens
        if requested_max_output_tokens is None:
            max_output_tokens = configured_max_output_tokens
        elif configured_max_output_tokens is None:
            max_output_tokens = requested_max_output_tokens
        else:
            max_output_tokens = min(
                requested_max_output_tokens,
                configured_max_output_tokens,
            )
        auxiliary_role_params = (
            config.planning_params,
            config.general_params,
            config.vlm_params,
            config.vlm_grounding_params,
            config.reflection_params,
        )
        role_params = (config.main_params, *auxiliary_role_params)
        if temperature is not None:
            for params in role_params:
                params.temperature = temperature
        if top_p is not None:
            for params in role_params:
                params.top_p = top_p

        # Reasoning checkpoints can otherwise consume the entire response cap
        # inside a private <think> span.  Keep role sessions independent while
        # allowing the RL recipe to reserve tokens for executable main-agent
        # code and parseable planner/reflection/finalizer answers.
        _configure_reasoning_roles(
            config,
            main_enable_thinking=self.config.main_enable_thinking,
            main_thinking_token_budget=self.config.main_thinking_token_budget,
            main_reasoning_budget=self.config.main_reasoning_budget,
            auxiliary_enable_thinking=self.config.auxiliary_enable_thinking,
            auxiliary_thinking_token_budget=(
                self.config.auxiliary_thinking_token_budget
            ),
            auxiliary_reasoning_budget=self.config.auxiliary_reasoning_budget,
        )
        # ``general_params`` is used by SpatialClaw's force-termination VLM,
        # which is an isolated, non-trainable session.  A reasoning checkpoint
        # can legitimately need more room here than planner/reflection calls
        # to finish its private span and emit the public boxed answer.  Keep
        # these controls separate so that extra finalizer tokens never enter
        # the exact main-agent history or policy loss.
        _configure_finalizer_role(
            config,
            enable_thinking=self.config.finalizer_enable_thinking,
            thinking_token_budget=self.config.finalizer_thinking_token_budget,
            reasoning_budget=self.config.finalizer_reasoning_budget,
        )
        if max_output_tokens is not None:
            config.main_params.max_tokens = max_output_tokens
        auxiliary_max_output_tokens = self.config.auxiliary_max_output_tokens
        for params in auxiliary_role_params:
            role_max_output_tokens = max_output_tokens
            if auxiliary_max_output_tokens is not None:
                role_max_output_tokens = (
                    auxiliary_max_output_tokens
                    if role_max_output_tokens is None
                    else min(role_max_output_tokens, auxiliary_max_output_tokens)
                )
            if role_max_output_tokens is not None:
                params.max_tokens = role_max_output_tokens
        if self.config.finalizer_max_output_tokens is not None:
            config.general_params.max_tokens = (
                self.config.finalizer_max_output_tokens
            )

        # Hard RL invariants. A context overflow is an incomplete rollout; it
        # must never be hidden by rewriting or removing prior policy tokens.
        for name, value in {
            "condense_errors": False,
            "max_llm_history_messages": -1,
            "max_llm_history_text_chars": -1,
        }.items():
            if hasattr(config, name):
                setattr(config, name, value)
        return config

    async def _materialize_inputs(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        run_metadata: dict[str, Any],
        session_dir: Path,
        spatial_config: Any,
    ) -> tuple[str, list[str], dict[str, Any]]:
        instruction, image_urls, video_urls = _extract_request_input(body)
        input_dir = session_dir / "request_media"

        def materialize() -> tuple[list[str], list[str]]:
            images = [
                _materialize_url(url, input_dir / f"image-{idx}{_suffix_for_url(url, '.png')}")
                for idx, url in enumerate(image_urls)
            ]
            videos = [
                _materialize_url(url, input_dir / f"video-{idx}{_suffix_for_url(url, '.mp4')}")
                for idx, url in enumerate(video_urls)
            ]
            return images, videos

        image_paths, video_paths = await asyncio.to_thread(materialize)
        ref_image_urls = list(run_metadata.get("ref_images") or [])
        if ref_image_urls:
            run_metadata["ref_images"] = await asyncio.to_thread(
                lambda: [
                    _materialize_url(
                        url,
                        input_dir
                        / f"ref-image-{idx}{_suffix_for_url(url, '.png')}",
                    )
                    for idx, url in enumerate(ref_image_urls)
                ]
            )
        frame_indices = list(run_metadata.get("frame_indices") or [])

        if video_paths:
            from spatial_agent.evals.base import extract_video_frames

            extracted_frame_counts: list[int] = []
            for index, video_path in enumerate(video_paths):
                cache_dir = session_dir / "video_frames" / str(index)
                frames, indices, fps, total = await asyncio.to_thread(
                    extract_video_frames,
                    video_path,
                    str(cache_dir),
                    getattr(spatial_config, "video_max_fps", None),
                    getattr(spatial_config, "video_frame_resize_short_edge", None),
                )
                minimum_frames = int(getattr(spatial_config, "num_key_frames", 0) or 0)
                retry_fps = _minimum_frame_retry_fps(
                    extracted_count=len(frames),
                    native_fps=float(fps),
                    total_native_frames=int(total),
                    minimum_frames=minimum_frames,
                )
                if retry_fps is not None:
                    retry_cache_dir = cache_dir / f"minimum_{minimum_frames}"
                    frames, indices, fps, total = await asyncio.to_thread(
                        extract_video_frames,
                        video_path,
                        str(retry_cache_dir),
                        retry_fps,
                        getattr(spatial_config, "video_frame_resize_short_edge", None),
                    )
                if minimum_frames > 0 and len(frames) < minimum_frames:
                    raise ValueError(
                        f"Video {video_path} produced only {len(frames)} frames, "
                        f"fewer than required num_key_frames={minimum_frames} "
                        f"(native_fps={fps}, total_native_frames={total})"
                    )
                image_paths.extend(frames)
                frame_indices.extend(indices)
                extracted_frame_counts.append(len(frames))
                run_metadata.setdefault("fps", fps)
                run_metadata.setdefault("total_video_frames", total)
                run_metadata.setdefault("duration_sec", total / fps if fps else 0.0)
            run_metadata["extracted_frame_counts"] = extracted_frame_counts
            run_metadata.setdefault("video_source", video_paths[0])
            if len(video_paths) > 1:
                run_metadata.setdefault("video_sources_per_video", video_paths)

        if not instruction:
            raise ValueError("SpatialClaw request has no text instruction")
        if not image_paths:
            raise ValueError("SpatialClaw request has no image or decodable video frames")
        run_metadata["frame_indices"] = frame_indices or list(range(len(image_paths)))
        return instruction, image_paths, run_metadata

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        async with self.sem:
            root = self._resolve_spatialclaw_root()
            install_spatialclaw_rl_hooks()

            run_metadata = _metadata(body)
            session_id = _session_id(run_metadata.get("session_id"))
            workspace_root = Path(self.config.workspace_root).expanduser().resolve()
            session_dir = workspace_root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            workflow = None
            turns = None
            try:
                spatial_config = self._build_spatialclaw_config(
                    root, run_metadata, session_dir, body
                )
                instruction, images, run_metadata = await self._materialize_inputs(
                    body, run_metadata, session_dir, spatial_config
                )

                source_videos = list(run_metadata.get("video_sources_per_video") or [])
                if not source_videos and run_metadata.get("video_source"):
                    source_videos = [str(run_metadata["video_source"])]
                start_capture(
                    session_id,
                    video_input_mode=self.config.video_input_mode,
                    source_videos=source_videos,
                )

                # SpatialClaw exposes selected video keyframes as image_url
                # parts to several independent sessions: planner, main agent,
                # reflection, force-termination VLM, and vlm.* tools. Preserve
                # those exact frames while asking NanoV3 to process every
                # video-origin session through the Conv3D path used by the 16K
                # SFT stage. Image-only requests retain independent-image
                # preprocessing.
                if run_metadata.get("video_source"):
                    _configure_video_role_preprocessing(spatial_config)

                from spatial_agent.workflow import SpatialAgentWorkflow

                workflow = SpatialAgentWorkflow(spatial_config)
                instrument_llm_client(workflow.llm_client)

                group_sizes = run_metadata.get("image_group_sizes") or []
                image_groups = None
                if group_sizes:
                    image_groups = []
                    offset = 0
                    for size in group_sizes:
                        size = int(size)
                        image_groups.append(images[offset : offset + size])
                        offset += size
                    if offset != len(images):
                        raise ValueError("image_group_sizes does not cover all request images")

                try:
                    result = await asyncio.wait_for(
                        workflow.arun(
                            instruction=instruction,
                            images=images,
                            answer=None,
                            session_id=session_id,
                            frame_indices=run_metadata.get("frame_indices"),
                            video_source=run_metadata.get("video_source"),
                            fps=run_metadata.get("fps"),
                            total_video_frames=run_metadata.get("total_video_frames"),
                            duration_sec=run_metadata.get("duration_sec"),
                            image_groups=image_groups,
                            frame_indices_groups=run_metadata.get("frame_indices_groups"),
                            fps_per_video=run_metadata.get("fps_per_video"),
                            total_frames_per_video=run_metadata.get("total_frames_per_video"),
                            duration_per_video=run_metadata.get("duration_per_video"),
                            video_names=run_metadata.get("video_names"),
                            video_sources_per_video=run_metadata.get("video_sources_per_video"),
                            ref_images=run_metadata.get("ref_images"),
                        ),
                        timeout=self.config.timeout,
                    )
                except asyncio.TimeoutError:
                    # A long, isolated reflection or force-termination VLM call
                    # must not discard main-policy turns that were already
                    # sampled and captured exactly. Return the real partial
                    # trajectory with an empty answer so verification assigns
                    # its normal failure reward. If no policy turn completed,
                    # preserve the infrastructure failure instead of creating
                    # an empty training sample.
                    turns, result = _finish_timed_out_capture(session_id)
                    print(
                        "[spatialclaw_agent] workflow timed out after "
                        f"{len(turns)} captured main turn(s); "
                        "returning the exact partial trajectory",
                        file=sys.stderr,
                        flush=True,
                    )
                if turns is None:
                    turns = finish_capture(session_id)
                _write_capture_manifest(
                    session_dir,
                    turns,
                    result,
                    video_input_mode=self.config.video_input_mode,
                )
            except Exception:
                # Uvicorn is launched as a managed Gym subprocess and its
                # traceback is not guaranteed to reach the NeMo RL driver.
                # Keep a per-session diagnostic beside the preserved media so
                # failed image/video rollouts remain actionable.
                error_traceback = traceback.format_exc()
                print(
                    f"[spatialclaw_agent] session {session_id} failed:\n"
                    f"{error_traceback}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    (session_dir / "error.traceback.txt").write_text(
                        error_traceback,
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                discard_capture(session_id)
                if not self.config.keep_workspaces:
                    shutil.rmtree(session_dir, ignore_errors=True)
                raise
            finally:
                if workflow is not None:
                    workflow.shutdown()

            output = [
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{index}",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=turn.content,
                            annotations=[],
                        )
                    ],
                    prompt_token_ids=turn.prompt_token_ids,
                    generation_token_ids=turn.generation_token_ids,
                    generation_log_probs=turn.generation_log_probs,
                    finish_reason=turn.finish_reason,
                    prompt_multimodal_content=turn.prompt_multimodal_content or None,
                    prompt_mm_processor_kwargs=turn.prompt_mm_processor_kwargs or None,
                )
                for index, turn in enumerate(turns)
            ]
            # SpatialClaw's aggregate usage includes isolated planner,
            # reflection, and VLM calls. Responses usage for this RL harness is
            # intentionally limited to the captured trainable policy calls.
            input_tokens = sum(len(turn.prompt_token_ids) for turn in turns)
            output_tokens = sum(len(turn.generation_token_ids) for turn in turns)
            final_answer = (result.get("final_answer") or {}).get("text", "")
            response = NeMoGymResponse(
                id=f"spatialclaw-{session_id}",
                created_at=int(time()),
                model=self.config.model_name or self.config.model_server.name,
                object="response",
                output=output,
                tool_choice=body.tool_choice,
                tools=body.tools,
                parallel_tool_calls=body.parallel_tool_calls,
                metadata={
                    "spatialclaw_final_answer": str(final_answer),
                    "spatialclaw_termination_reason": str(result.get("termination_reason") or ""),
                    "spatialclaw_turns": str(len(turns)),
                    "spatialclaw_video_input_mode": self.config.video_input_mode,
                },
                usage=NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens=output_tokens,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=input_tokens + output_tokens,
                ),
            )
            if not self.config.keep_workspaces:
                shutil.rmtree(session_dir, ignore_errors=True)
            return response

    async def run(
        self,
        request: Request,
        body: SpatialClawAgentRunRequest = Body(),
    ) -> SpatialClawAgentVerifyResponse:
        cookies = request.cookies
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        agent_json = await get_response_json(agent_response)

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": agent_json},
            cookies=cookies,
        )
        await raise_for_status(verify)
        verify_json = await get_response_json(verify)
        metadata = agent_json.get("metadata") or {}
        return SpatialClawAgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": int(metadata.get("spatialclaw_turns", 0) or 0),
                "termination_reason": metadata.get("spatialclaw_termination_reason"),
            }
        )

    async def aggregate_metrics(
        self, body: AggregateMetricsRequest = Body()
    ) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SpatialClawAgent.run_webserver()
