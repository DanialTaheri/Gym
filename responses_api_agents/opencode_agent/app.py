# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import contextlib
import copy
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import ssl
import sys
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.chat_streaming import sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.opencode_agent.observability import append_opencode_turns, scope_opencode_trajectory
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode
from responses_api_agents.opencode_agent.token_tracking import (
    TrackedCall,
    attach_parent_tokens,
    chain_output_items,
    drop_echoed_reasoning,
    record_call,
)


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_INTERNAL_TRAJECTORY_KEY = "_ng_trajectory"


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    """Convert OpenCode's Date.now()-based epoch milliseconds to seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return float(value) / 1000


def _parse_opencode_session(
    db_path: Path, fallback_invocation_id: str, trajectory: Optional[TrajectoryRecord] = None
) -> AgentObservationBundle:
    """Read OpenCode's persisted session tree before its workspace is removed."""
    if not db_path.is_file():
        return AgentObservationBundle(
            source="opencode",
            records=[AgentInvocation(invocation_id=fallback_invocation_id)],
            gaps=[
                ObservationGap(code="agent_artifact_unavailable"),
                ObservationGap(code="agent_transcript_unavailable"),
                ObservationGap(code="model_call_ownership_unavailable"),
            ],
        )

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        session_rows = con.execute(
            "select id, parent_id, time_created from session order by time_created, id"
        ).fetchall()
        message_rows = con.execute(
            "select id, session_id, data, time_created from message order by time_created, id"
        ).fetchall()
        part_rows = con.execute(
            "select id, message_id, session_id, data, time_created from part order by time_created, id"
        ).fetchall()
    finally:
        con.close()

    messages = {row["id"]: _load_json(row["data"]) for row in message_rows}
    message_sessions = {row["id"]: row["session_id"] for row in message_rows}
    conversations: dict[str, list[Any]] = {row["id"]: [] for row in session_rows}
    invocation_status: dict[str, str] = {row["id"]: "unknown" for row in session_rows}
    tools: list[ToolCallObservation] = []
    child_tools: dict[str, set[str]] = {}
    child_status: dict[str, str] = {}
    compaction_parts: list[tuple[str, str, float | None, dict[str, Any]]] = []
    gaps: list[ObservationGap] = []
    summary_text: dict[str, list[str]] = {}
    summaries_by_parent: dict[str, list[str]] = {}
    first_item_id_by_message: dict[tuple[str, str], str] = {}

    for row in message_rows:
        message = messages[row["id"]]
        session_id = row["session_id"]
        if not isinstance(session_id, str) or session_id not in invocation_status:
            gaps.append(ObservationGap(code="agent_artifact_record_unowned", detail=row["id"]))
        elif message.get("role") == "assistant":
            if isinstance(message.get("error"), dict):
                invocation_status[session_id] = "failed"
            message_time = message.get("time") if isinstance(message.get("time"), dict) else {}
            if invocation_status[session_id] != "failed" and _milliseconds(message_time.get("completed")) is not None:
                invocation_status[session_id] = "completed"
        if message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault(parent_id, []).append(row["id"])

    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")

        if ptype == "step-finish":
            continue

        text = part.get("text")
        if ptype == "text" and isinstance(text, str) and text.strip():
            if message_id in summary_text:
                summary_text[message_id].append(text)
            if role == "user" and part.get("ignored") is not True:
                conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role == "assistant":
                item = NeMoGymResponseOutputMessage(
                    id=row["id"],
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
                conversation.append(item)
                first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "reasoning" and role == "assistant" and isinstance(text, str) and text.strip():
            conversation.append(
                NeMoGymResponseReasoningItem(
                    id=row["id"],
                    summary=[NeMoGymSummary(type="summary_text", text=text)],
                )
            )
            first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "tool" and role == "assistant":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            native_call_id = part.get("callID")
            observed_call_id = native_call_id if isinstance(native_call_id, str) and native_call_id else None
            call_id = observed_call_id or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            native_status = state.get("status")
            response_status = "completed" if native_status == "completed" else "incomplete"
            call = NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id=call_id,
                name=part.get("tool", ""),
                type="function_call",
                id=call_id,
                status=response_status,
            )
            conversation.append(call)
            first_item_id_by_message.setdefault((session_id, message_id), call_id)
            native_time = state.get("time") if isinstance(state.get("time"), dict) else {}
            # OpenCode retains raw output in SQLite but substitutes this literal in later model inputs after pruning.
            if native_status == "completed" and native_time.get("compacted") is not None:
                observed_tool_output = "[Old tool result content cleared]"
            else:
                observed_tool_output = state.get("output") if state.get("output") is not None else state.get("error")
            if observed_tool_output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(observed_tool_output),
                    status=response_status,
                )
                conversation.append(result)

            native_start = native_time.get("start")
            native_end = native_time.get("end")
            valid_interval = (
                isinstance(native_start, (int, float))
                and not isinstance(native_start, bool)
                and isinstance(native_end, (int, float))
                and not isinstance(native_end, bool)
                and native_end >= native_start
            )
            started_at = _milliseconds(native_start) if valid_interval else None
            completed_at = _milliseconds(native_end) if valid_interval else None
            duration_ms = float(native_end - native_start) if valid_interval else None
            status = {
                "completed": "completed",
                "error": "failed",
                "running": "incomplete",
                "pending": "incomplete",
            }.get(native_status, "unknown")
            if observed_call_id is not None:
                tools.append(
                    ToolCallObservation(
                        invocation_id=session_id,
                        tool_call_id=observed_call_id,
                        tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        timing_source="artifact" if started_at is not None else None,
                        status=status,
                        error_type="tool_error" if native_status == "error" else None,
                    )
                )
            else:
                gaps.append(
                    ObservationGap(
                        code="tool_call_identity_unavailable",
                        invocation_id=session_id,
                        detail=row["id"],
                    )
                )
            if observed_call_id is not None and (
                started_at is None or (native_status in {"completed", "error"} and completed_at is None)
            ):
                gaps.append(
                    ObservationGap(
                        code="tool_timing_unavailable",
                        invocation_id=session_id,
                        detail=observed_call_id,
                    )
                )
            metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            if not metadata and isinstance(part.get("metadata"), dict):
                metadata = part["metadata"]
            child_id = metadata.get("sessionId")
            if isinstance(child_id, str) and observed_call_id is not None:
                child_tools.setdefault(child_id, set()).add(observed_call_id)
                child_status[child_id] = status
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get(message_id, [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
                )
            )
        trigger = "overflow" if part.get("overflow") is True else "automatic" if part.get("auto") is True else "manual"
        tail_start_id = part.get("tail_start_id") if isinstance(part.get("tail_start_id"), str) else None
        first_kept_item_id = (
            first_item_id_by_message.get((session_id, tail_start_id)) if tail_start_id is not None else None
        )
        compactions.append(
            ContextCompactionObservation(
                invocation_id=session_id,
                observed_at=observed_at,
                trigger=trigger,
                outcome="completed" if summary else "unknown",
                summary=summary,
                first_kept_item_id=first_kept_item_id,
            )
        )
        if tail_start_id is not None and first_kept_item_id is None:
            gaps.append(
                ObservationGap(
                    code="compaction_first_kept_item_unavailable",
                    invocation_id=session_id,
                    detail=tail_start_id,
                )
            )
        if not summary:
            gaps.append(ObservationGap(code="compaction_summary_unavailable", invocation_id=session_id))
        gaps.append(ObservationGap(code="compaction_token_counts_unavailable", invocation_id=session_id))
        gaps.append(
            ObservationGap(
                code="compaction_model_call_boundary_unavailable",
                invocation_id=session_id,
            )
        )

    session_ids = {row["id"] for row in session_rows}
    invocations = []
    for row in session_rows:
        invocation_id = row["id"]
        parent_id = row["parent_id"]
        spawn_candidates = child_tools.get(invocation_id, set())
        if parent_id is not None and parent_id not in session_ids:
            gaps.append(
                ObservationGap(
                    code="subagent_parent_unavailable",
                    invocation_id=invocation_id,
                    detail=parent_id,
                )
            )
        if len(spawn_candidates) > 1:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_ambiguous",
                    invocation_id=invocation_id,
                )
            )
        elif parent_id is not None and not spawn_candidates:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_tool_unavailable",
                    invocation_id=invocation_id,
                )
            )
        invocations.append(
            AgentInvocation(
                invocation_id=invocation_id,
                parent_invocation_id=parent_id,
                spawned_by_tool_call_id=next(iter(spawn_candidates)) if len(spawn_candidates) == 1 else None,
                status=(
                    invocation_status.get(invocation_id, "unknown")
                    if invocation_status.get(invocation_id, "unknown") != "unknown"
                    else child_status.get(invocation_id, "unknown")
                ),
                conversation=conversations.get(invocation_id, []),
            )
        )
    if not invocations:
        invocations = [AgentInvocation(invocation_id=fallback_invocation_id)]
        gaps.append(ObservationGap(code="agent_transcript_unavailable"))

    if trajectory is not None:
        append_opencode_turns(trajectory, session_ids, message_rows, part_rows)

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


def parse_opencode_session(db_path: Path) -> tuple[list[Any], dict[str, int]]:
    """Convert an OpenCode session database into the existing Gym response shape."""
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    if not db_path.is_file():
        return output_items, {"input_tokens": 0, "output_tokens": 0}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        roles = {row["id"]: _load_json(row["data"]).get("role") for row in con.execute("select id, data from message")}
        rows = con.execute("select message_id, data from part order by time_created").fetchall()
    finally:
        con.close()

    for row in rows:
        part = _load_json(row["data"])
        ptype = part.get("type")
        if ptype == "step-finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)
        elif roles.get(row["message_id"]) == "assistant" and ptype == "text" and (part.get("text") or "").strip():
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=part["text"], annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif roles.get(row["message_id"]) == "assistant" and ptype == "tool":
            state = part.get("state") or {}
            call_id = part.get("callID") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            if state.get("output") is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(state["output"]),
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}


def _extract_images(body_input) -> list[tuple[str, bytes]]:
    """(suffix, bytes) of the data-URL images in the last user message, in order."""
    for item in reversed(list(body_input)):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role != "user":
            continue
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        images = []
        for part in content if isinstance(content, list) else []:
            data = part if isinstance(part, dict) else part.model_dump()
            url = data.get("image_url") if data.get("type") == "input_image" else None
            match = re.match(r"data:(image/[a-z]+);base64,(.*)", url or "", re.S)
            if match and match.group(1) in _IMAGE_SUFFIXES:
                images.append((_IMAGE_SUFFIXES[match.group(1)], base64.b64decode(match.group(2))))
        return images
    return []


# Runs as `sh -c SCRIPT sh <work_dir> <keep_dirs> <command...>` in a fresh user+mount namespace:
# the run's own <work_dir>/tmp becomes /tmp, then the colon-separated <keep_dirs> (top-level
# /tmp entries the run needs: Gym, its venvs, OpenCode) are bound back from the host's /tmp.
_PRIVATE_TMP_SCRIPT = """set -e
work=$1; keep=$2; shift 2
start=$(pwd)
mkdir -p "$work/.host-tmp" "$work/tmp"
mount --rbind /tmp "$work/.host-tmp"
cd "$work"
mount --bind "$work/tmp" /tmp
IFS=:
for dir in $keep; do
  mkdir -p "$dir"
  mount --rbind ".host-tmp/${dir#/tmp/}" "$dir"
done
unset IFS
cd "$start"
exec "$@"
"""


_REMOTE_SESSION_TASK = None


def _remote_session_task():
    """The Ray task that runs one OpenCode session on a session node (created once).

    The body is defined inside this function so Ray pickles it by value: workers on other
    nodes need not import this module. It recreates the run dir from `inputs` (the image and
    OpenCode config), runs OpenCode in its own process group, returns the session database
    files, and removes the run dir.
    """
    global _REMOTE_SESSION_TASK
    if _REMOTE_SESSION_TASK is not None:
        return _REMOTE_SESSION_TASK
    import ray

    def run_session(cmd, work_dir, project_dir, env, inputs, timeout):
        import os
        import shutil
        import signal
        import subprocess
        from pathlib import Path

        for path, data in inputs.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        data_home = Path(env["XDG_DATA_HOME"])
        data_home.mkdir(parents=True, exist_ok=True)
        timed_out = False
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=project_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                _, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                _, stderr = proc.communicate()
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            outputs = {
                str(path): path.read_bytes()
                for path in (data_home / "opencode").glob("opencode.db*")
                if path.is_file()
            }
            return proc.returncode, stderr[-4000:], timed_out, outputs
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    _REMOTE_SESSION_TASK = ray.remote(num_cpus=0, max_retries=0)(run_session)
    return _REMOTE_SESSION_TASK


def _kill_process_group(pgid: int) -> None:
    """SIGKILL a run's whole process group (OpenCode and anything the model left running)."""
    # Never our own group: killpg(0) or our own pgid would kill this server.
    if not isinstance(pgid, int) or pgid <= 1 or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _tmp_roots(paths: list[Optional[str]]) -> list[str]:
    """The top-level /tmp/<name> directory of each path that lives under /tmp, deduplicated."""
    roots = []
    for path in paths:
        parts = Path(os.path.realpath(path)).parts if path else ()
        root = str(Path(*parts[:3])) if len(parts) > 2 and parts[:2] == ("/", "tmp") else None
        if root and root not in roots:
            roots.append(root)
    return roots


# Session-sandbox environment: the model endpoint and where OpenCode keeps its session database. Nothing else
# of this agent's environment (which holds credentials) reaches the sandbox.
_SANDBOX_SESSION_ENV_KEYS = ("XDG_DATA_HOME", "OPENAI_BASE_URL", "OPENAI_API_KEY")
_SANDBOX_DEFAULT_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root"}
_SANDBOX_MARK = "@@ng-opencode@@"


class OpenCodeSessionSandboxConfig(BaseModel):
    """Run each OpenCode session in its own sandbox from a nemo_gym.sandbox provider (e.g. OpenSandbox).

    The model's tool code then runs in an isolated container with its own CPU/memory limits instead of on
    this agent's hosts. The session's run dir (OpenCode config, task images) is uploaded to the same
    absolute paths in the sandbox, OpenCode runs there and calls the model through this agent as usual,
    and the session database is downloaded back. Each sandbox is deleted after its session.
    """

    model_config = ConfigDict(extra="forbid")

    # Single-key nemo_gym.sandbox provider config, e.g. {"opensandbox": {"connection": {...}, "operations": {...}}}.
    # Keep credentials out of it: see provider_secret_env.
    provider: dict[str, Any]
    # Fields of the provider's `connection` block read from environment variables when the provider is
    # created, e.g. {"domain": "OPENSANDBOX_DOMAIN", "api_key": "OPENSANDBOX_API_KEY"}, so they never
    # appear in configs or logs.
    provider_secret_env: dict[str, str] = Field(default_factory=dict)
    # SandboxSpec fields for every session sandbox: image, resources, metadata, provider_options (for
    # example {"snapshot_id": ...}), ready_timeout_s, ttl_s (default: timeout + setup_timeout + 600 s).
    spec: dict[str, Any] = Field(default_factory=dict)
    # The OpenCode command inside the sandbox (used instead of `command`).
    command: str = "opencode"
    # Shell command run in each sandbox before OpenCode, e.g. installing the toolkit when the image or
    # snapshot does not have it.
    setup_command: Optional[str] = None
    setup_timeout: int = 900
    # Environment for OpenCode in the sandbox, on top of the model endpoint, XDG_DATA_HOME and this
    # agent's `env` (PATH and HOME default to the image's usual values).
    env: dict[str, str] = Field(default_factory=dict)
    # PEM file this agent process trusts in addition to its default CAs, e.g. a provider server's
    # self-signed certificate. Applied by pointing SSL_CERT_FILE at a merged bundle.
    ca_file: Optional[str] = None
    # At most this many sessions create and prepare their sandbox at once (the rest wait): a burst of
    # hundreds of concurrent creates, each polled by the SDK until ready, starves this agent's event loop
    # and every OpenSandbox request then times out. 0 = no limit.
    max_concurrent_starts: int = 64


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/opencode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = True
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    opencode_version: Optional[str] = None
    # Multimodal tasks: write the last user message's data-URL images into the project dir as
    # image_<i>.<ext> and attach them with -f. `vision` declares a generated (model_server)
    # model entry image-capable, so OpenCode sends attachments and `read` returns image files.
    attach_images: bool = False
    vision: bool = False
    # Give each run a private /tmp (unshare --user --mount; needs unprivileged user namespaces).
    # Models write scratch files to fixed /tmp names, which concurrent runs would share.
    private_tmp: bool = False
    # RL training: route OpenCode's model calls through this agent, which pins each call's
    # prompt to the tokens sampled before and returns the calls' token IDs and logprobs as the
    # rollout output (see token_tracking.py). Requires model_server and a single server worker.
    track_token_ids: bool = False
    # Spread OpenCode sessions over this many cluster nodes (this server's node first, then other
    # Ray nodes), each session a Ray task pinned round-robin to one of them, so the session
    # count is not bound by one host's CPUs and RAM. A session runs entirely on its node:
    # its run dir (workspace_root, best node-local) is recreated there, and only the small
    # OpenCode session database comes back. Needs Ray and identical paths on every node
    # (OpenCode, this venv). 1 = run locally as a subprocess.
    session_nodes: int = 1
    # Ray actor class names (e.g. "MegatronPolicyWorker") whose nodes never host sessions: RL training
    # nodes need their host memory while computing logprobs and training (sessions there hit host OOM).
    # Looked up once, when the session nodes are first chosen; empty = every alive node is eligible.
    session_exclude_actor_classes: list[str] = Field(default_factory=list)
    # Run each session in its own sandbox instead of on a cluster node (see OpenCodeSessionSandboxConfig).
    # Takes the place of session_nodes and private_tmp: a sandbox is private by construction.
    session_sandbox: Optional[OpenCodeSessionSandboxConfig] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class OpenCodeAgent(SimpleResponsesAPIAgent):
    """Runs the CLI (opencode run --format=json)"""

    config: OpenCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._tracked_runs: dict[str, list[TrackedCall]] = {}
        self._session_nodes: Optional[list[str]] = None
        self._next_session_node = 0
        self._sandbox_provider: Any = None
        if self.config.session_sandbox is not None and self.config.session_nodes > 1:
            raise ValueError("session_sandbox and session_nodes > 1 are alternatives; set only one")
        if self.config.track_token_ids and self.config.model_server is None:
            raise ValueError("track_token_ids requires model_server")
        if self.config.track_token_ids and (self.config.num_workers or 1) > 1:
            raise ValueError("track_token_ids keeps per-run state in memory and needs num_workers <= 1")
        ensure_opencode(self.config.opencode_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("opencode command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/ng-tok/{run_key}/v1/chat/completions")(self.tracked_chat_completions)
        return app

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None, run_key: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        if run_key is not None:
            server_config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.name)
            return f"{self.server_client._build_server_base_url(server_config)}/ng-tok/{run_key}/v1"
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def tracked_chat_completions(self, run_key: str, request: Request) -> Response:
        """OpenCode's model endpoint under track_token_ids: forward to the policy, keep the tokens."""
        body = await request.json()
        include_usage = False
        stream = body.get("stream") is True
        if stream:
            body, include_usage = sanitize_streaming_chat_body(body)
        calls = self._tracked_runs.get(run_key)
        parent, new_images = attach_parent_tokens(body.get("messages") or [], calls or [])
        drop_echoed_reasoning(body.get("messages") or [])
        model_resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json=body,
            cookies=request.cookies,
        )
        if not model_resp.ok:
            return Response(
                content=await model_resp.read(),
                status_code=model_resp.status,
                media_type=model_resp.headers.get("content-type"),
            )
        completion = await get_response_json(model_resp)
        message = completion["choices"][0]["message"]
        if calls is not None:
            record_call(calls, parent, new_images, message)
        else:
            record_call([], parent, new_images, message)  # a late call after the run ended: just strip tokens
        if stream:
            return StreamingResponse(
                synthesize_chat_completion_sse(completion, include_usage), media_type="text/event-stream"
            )
        return JSONResponse(completion)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_opencode_config(
        self, rollout_id: Optional[str] = None, run_key: Optional[str] = None
    ) -> dict[str, Any]:
        config = self._deep_merge({}, copy.deepcopy(self.config.opencode_config))
        if self.config.model_server:
            providers = config.setdefault("provider", {})
            nemo = providers.setdefault("nemo", {"npm": "@ai-sdk/openai-compatible"})
            nemo.setdefault("options", {}).update(
                {
                    "baseURL": self._resolve_model_base_url(rollout_id, run_key),
                    "apiKey": "EMPTY",
                }  # pragma: allowlist secret
            )
            model = nemo.setdefault("models", {}).get(self.config.model, {})
            self._deep_merge(
                model,
                {
                    "name": self.config.model,
                    "interleaved": {"field": "reasoning"},
                    "limit": {"context": self.config.context_window, "output": self.config.max_output_tokens},
                    **(
                        {"attachment": True, "modalities": {"input": ["text", "image"], "output": ["text"]}}
                        if self.config.vision
                        else {}
                    ),
                },
            )
            nemo["models"] = {self.config.model: model}
        return config

    def _write_opencode_config(
        self, work_dir: Path, rollout_id: Optional[str] = None, run_key: Optional[str] = None
    ) -> None:
        config = self._build_opencode_config(rollout_id, run_key)
        if not config:
            return
        (work_dir / "opencode.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, rollout_id: Optional[str] = None, run_key: Optional[str] = None) -> dict[str, str]:
        env = {**os.environ, "XDG_DATA_HOME": data_home}
        # OpenCode also writes its config/cache/state dirs; keep them in the run dir too
        # (clusters where $HOME is read-only).
        for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
            env[name] = str(Path(data_home).parent / f".opencode-{name.split('_')[1].lower()}")
        base_url = (
            self._resolve_model_base_url(rollout_id, run_key)
            if self.config.model_server
            else self.config.openai_base_url
        )
        api_key = "EMPTY" if self.config.model_server else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        if self.config.model_server is not None:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = "EMPTY"  # pragma: allowlist secret
        return env

    async def _run_opencode(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
        trajectory: Optional[TrajectoryRecord] = None,
        images: Optional[list[tuple[str, bytes]]] = None,
    ) -> tuple[list[Any], dict[str, int], str, AgentObservationBundle]:
        """Run one headless OpenCode session and read its persisted artifact."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".opencode-data"
        data_home.mkdir(parents=True, exist_ok=True)
        run_key = uuid4().hex if self.config.track_token_ids else None
        if run_key is not None:
            self._tracked_runs[run_key] = []
        self._write_opencode_config(project_dir, rollout_id, run_key)
        env = self._env(str(data_home), rollout_id, run_key)
        files = []
        for index, (suffix, data) in enumerate(images or []):
            path = project_dir / f"image_{index}{suffix}"
            path.write_bytes(data)
            files.append(str(path.resolve()))

        opencode_args = ["run", "-m", self._effective_model(), "--dir", str(project_dir)]
        if self.config.thinking:
            opencode_args.append("--thinking")
        opencode_args.extend(self.config.extra_args)
        opencode_args.append(prompt)
        # Repeated -f (an array flag), after the positional prompt so it cannot absorb it.
        opencode_args.extend(arg for path in files for arg in ("-f", path))
        cmd = [*self.config.command_parts, *opencode_args]
        if self.config.private_tmp and self.config.session_sandbox is None:
            keep = _tmp_roots(
                [str(work_dir), str(project_dir), str(Path(__file__)), sys.prefix, sys.base_prefix, sys.executable]
                + [shutil.which(self.config.command_parts[0]) if self.config.command_parts else None]
            )
            cmd = [
                *("unshare", "--user", "--map-current-user", "--mount", "sh", "-c", _PRIVATE_TMP_SCRIPT, "sh"),
                str(work_dir.resolve()),
                ":".join(keep),
                *cmd,
            ]

        try:
            if self.config.session_sandbox is not None:
                returncode, stderr, timed_out = await self._run_sandbox_session(
                    opencode_args, work_dir, project_dir, env
                )
            elif self.config.session_nodes > 1:
                returncode, stderr, timed_out = await self._run_remote_session(cmd, work_dir, project_dir, env)
            else:
                returncode, stderr, timed_out = await self._run_local_session(cmd, project_dir, env)
            if timed_out:
                LOG.warning("opencode timed out after %ds", self.config.timeout)
            if returncode not in (0, None):
                LOG.warning("opencode exited %d: %s", returncode, stderr.decode(errors="replace")[:500])

            db_path = data_home / "opencode" / "opencode.db"
            invocation_id = rollout_id or f"opencode-{uuid4().hex}"
            output_items, usage = (
                ([], {"input_tokens": 0, "output_tokens": 0}) if timed_out else parse_opencode_session(db_path)
            )
            if run_key is not None:
                # Training output: the model calls themselves, with their token IDs. A timed-out
                # run keeps the calls it made (its reward is 0).
                output_items = chain_output_items(self._tracked_runs[run_key], len(images or []))
            observations = AgentObservationBundle(source="opencode")
            if collect_observations:
                try:
                    observations = _parse_opencode_session(db_path, invocation_id, trajectory)
                except Exception:
                    LOG.exception("failed to read OpenCode session artifact")
                    if trajectory is not None:
                        trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
                    observations = AgentObservationBundle(
                        source="opencode",
                        records=[AgentInvocation(invocation_id=invocation_id)],
                        gaps=[
                            ObservationGap(code="agent_artifact_unavailable"),
                            ObservationGap(code="agent_transcript_unavailable"),
                            ObservationGap(code="model_call_ownership_unavailable"),
                        ],
                    )
            run_status = "incomplete" if timed_out else "completed" if returncode == 0 else "failed"
            for invocation in observations.records:
                if not isinstance(invocation, AgentInvocation):
                    continue
                if invocation.parent_invocation_id is None:
                    invocation.status = run_status
            if timed_out:
                observations.gaps.append(ObservationGap(code="agent_run_timeout"))
            return output_items, usage, self.config.model, observations
        finally:
            if run_key is not None:
                self._tracked_runs.pop(run_key, None)
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _run_local_session(
        self, cmd: list[str], project_dir: Path, env: dict[str, str]
    ) -> tuple[Optional[int], bytes, bool]:
        """Run one OpenCode session as a local subprocess: (returncode, stderr, timed_out)."""
        timed_out = False
        # Own process group: the model's shell commands (python scripts, background jobs)
        # are OpenCode's descendants and must die with the run, or they pile up on the host.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid)
            _, stderr = await proc.communicate()
            timed_out = True
        finally:
            _kill_process_group(proc.pid)
        return proc.returncode, stderr, timed_out

    def _excluded_session_node_ids(self) -> set[str]:
        """Nodes running an alive actor of a session_exclude_actor_classes class."""
        if not self.config.session_exclude_actor_classes:
            return set()
        from ray.util.state import list_actors

        excluded = set()
        for class_name in self.config.session_exclude_actor_classes:
            actors = list_actors(
                filters=[("class_name", "=", class_name), ("state", "=", "ALIVE")],
                # Ray's state API rejects limits above 10,000 (RAY_MAX_LIMIT_FROM_API_SERVER).
                limit=10_000,
                raise_on_missing_output=False,
            )
            excluded.update(actor.node_id for actor in actors if actor.node_id)
        return excluded

    def _session_node_ids(self) -> list[str]:
        """This server's Ray node, then other alive nodes (sorted by IP), up to session_nodes,
        skipping nodes that run a session_exclude_actor_classes actor."""
        if self._session_nodes is None:
            import ray

            excluded = self._excluded_session_node_ids()
            own = ray.get_runtime_context().get_node_id()
            others = sorted(
                (node for node in ray.nodes() if node["Alive"] and node["NodeID"] != own),
                key=lambda node: node["NodeManagerAddress"],
            )
            candidates = [own] + [node["NodeID"] for node in others]
            nodes = [node_id for node_id in candidates if node_id not in excluded][: self.config.session_nodes]
            if not nodes:
                raise RuntimeError(
                    f"No Ray node left for OpenCode sessions after excluding nodes running "
                    f"{self.config.session_exclude_actor_classes} ({len(excluded)} nodes)"
                )
            self._session_nodes = nodes
            LOG.warning(
                "OpenCode sessions spread over %d Ray nodes (%d excluded for %s)",
                len(nodes),
                len(excluded),
                self.config.session_exclude_actor_classes,
            )
        return self._session_nodes

    async def _run_remote_session(
        self, cmd: list[str], work_dir: Path, project_dir: Path, env: dict[str, str]
    ) -> tuple[Optional[int], bytes, bool]:
        """Run one OpenCode session on a session node; copy its session database back here."""
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = self._session_node_ids()
        node_id = nodes[self._next_session_node % len(nodes)]
        self._next_session_node += 1
        inputs = {str(path): path.read_bytes() for path in work_dir.rglob("*") if path.is_file()}
        task = _remote_session_task().options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=True)
        )
        returncode, stderr, timed_out, outputs = await task.remote(
            cmd, str(work_dir), str(project_dir), env, inputs, float(self.config.timeout)
        )
        for path, data in outputs.items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)
        return returncode, stderr, timed_out

    def _trust_ca_file(self, ca_file: str) -> None:
        """Make this process trust ``ca_file`` on top of its current CAs (SSL_CERT_FILE -> merged bundle)."""
        try:
            import certifi

            certifi_bundle = certifi.where()
        except ModuleNotFoundError:
            certifi_bundle = None
        sources = [os.environ.get("SSL_CERT_FILE"), ssl.get_default_verify_paths().cafile, certifi_bundle, ca_file]
        bundle = []
        for source in dict.fromkeys(source for source in sources if source):
            if os.path.isfile(source):
                bundle.append(Path(source).read_text())
        if not os.path.isfile(ca_file):
            raise FileNotFoundError(f"session_sandbox.ca_file not found: {ca_file}")
        target = Path(self.config.workspace_root).expanduser() / f"ca_bundle_{os.getpid()}.pem"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(bundle))
        os.environ["SSL_CERT_FILE"] = str(target)

    def _session_sandbox_provider(self) -> Any:
        """The sandbox provider shared by all session sandboxes (created on first use)."""
        if self._sandbox_provider is None:
            from nemo_gym.sandbox import create_provider

            sandbox_config = self.config.session_sandbox
            provider_config = copy.deepcopy(sandbox_config.provider)
            if len(provider_config) != 1:
                raise ValueError("session_sandbox.provider must contain exactly one provider name")
            provider_kwargs = next(iter(provider_config.values()))
            connection = provider_kwargs.setdefault("connection", {})
            for field_name, env_name in sandbox_config.provider_secret_env.items():
                value = os.environ.get(env_name)
                if not value:
                    raise RuntimeError(f"session_sandbox: environment variable {env_name} is not set")
                connection[field_name] = value
            if sandbox_config.ca_file:
                self._trust_ca_file(sandbox_config.ca_file)
            self._sandbox_provider = create_provider(provider_config)
        return self._sandbox_provider

    def _sandbox_start_limit(self) -> Any:
        """Context manager admitting at most session_sandbox.max_concurrent_starts sandbox starts at once."""
        limit = self.config.session_sandbox.max_concurrent_starts
        if limit <= 0:
            return contextlib.nullcontext()
        if getattr(self, "_sandbox_start_sem", None) is None:
            self._sandbox_start_sem = Semaphore(limit)
        return self._sandbox_start_sem

    async def _run_sandbox_session(
        self, opencode_args: list[str], work_dir: Path, project_dir: Path, env: dict[str, str]
    ) -> tuple[Optional[int], bytes, bool]:
        """Run one OpenCode session in its own sandbox; copy its session database back here.

        A sandbox infrastructure failure is reported like a failed OpenCode run (no answer), not raised:
        one bad sandbox must not abort the rollout batch.
        """
        from nemo_gym.sandbox import SandboxSpec

        sandbox_config = self.config.session_sandbox
        provider = self._session_sandbox_provider()
        spec = SandboxSpec(
            **{"ttl_s": self.config.timeout + sandbox_config.setup_timeout + 600, **sandbox_config.spec}
        )
        sandbox_env = {**_SANDBOX_DEFAULT_ENV, **{key: env[key] for key in _SANDBOX_SESSION_ENV_KEYS if env.get(key)}}
        sandbox_env.update({key: value for key, value in self.config.env.items() if value})
        sandbox_env.update(sandbox_config.env)
        data_home = sandbox_env["XDG_DATA_HOME"]
        inputs = [path for path in work_dir.rglob("*") if path.is_file()]
        directories = {str(work_dir), str(project_dir), data_home, *(str(path.parent) for path in inputs)}
        stderr_path = f"{work_dir}/.opencode-stderr"
        handle = None
        try:
            async with self._sandbox_start_limit():
                handle = await provider.create(spec)
                prepare = "mkdir -p " + " ".join(shlex.quote(directory) for directory in sorted(directories))
                if sandbox_config.setup_command:
                    prepare = f"{prepare} && {sandbox_config.setup_command}"
                prepared = await provider.exec(handle, prepare, timeout_s=sandbox_config.setup_timeout)
                if prepared.return_code != 0:
                    detail = f"{prepared.stdout or ''}{prepared.stderr or ''}"[-2000:]
                    return None, f"sandbox setup failed ({prepared.return_code}): {detail}".encode(), False
                for path in inputs:
                    await provider.upload_file(handle, path, str(path))
            command = [*shlex.split(sandbox_config.command), *opencode_args]
            script = (
                f"cd {shlex.quote(str(project_dir))} && "
                f"timeout -k 10 {self.config.timeout} {shlex.join(command)} >/dev/null 2>{shlex.quote(stderr_path)}; "
                f"rc=$?; tail -c 4000 {shlex.quote(stderr_path)}; echo; echo {_SANDBOX_MARK}rc=$rc; "
                f"for f in {shlex.quote(data_home)}/opencode/opencode.db*; "
                f'do [ -f "$f" ] && echo "{_SANDBOX_MARK}db=$f"; done; exit 0'
            )
            result = await provider.exec(handle, script, env=sandbox_env, timeout_s=self.config.timeout + 120)
            output = f"{result.stdout or ''}\n{result.stderr or ''}"
            returncode = None
            databases = []
            stderr_lines = []
            for line in output.splitlines():
                if line.startswith(f"{_SANDBOX_MARK}rc="):
                    returncode = int(line.removeprefix(f"{_SANDBOX_MARK}rc="))
                elif line.startswith(f"{_SANDBOX_MARK}db="):
                    databases.append(line.removeprefix(f"{_SANDBOX_MARK}db="))
                else:
                    stderr_lines.append(line)
            for database in databases:
                await provider.download_file(handle, database, Path(database))
            # timeout(1) exits 124 when it stopped the command, 137 when it had to SIGKILL it.
            return returncode, "\n".join(stderr_lines).encode()[-4000:], returncode in (124, 137)
        except Exception as e:
            LOG.warning("OpenCode session sandbox failed: %r", e)
            return None, f"sandbox session failed: {e!r}".encode()[-4000:], False
        finally:
            if handle is not None:
                try:
                    await provider.close(handle)
                except Exception as e:
                    LOG.warning("failed to delete session sandbox %s: %r", handle.sandbox_id, e)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = user_message if system_prompt is None else f"{system_prompt}\n\n{user_message}"
        trajectory = (
            TrajectoryRecord(task_id="unscoped", rollout_id=rollout_id or "unscoped") if collect_observations else None
        )
        images = _extract_images(body.input) if self.config.attach_images else []

        output_items, usage, model_name, observations = await self._run_opencode(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
            trajectory=trajectory,
            images=images,
        )
        if collect_observations:
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))

        if collect_observations:
            root = next(
                (
                    record
                    for record in observations.records
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
                ),
                None,
            )
            if root is not None and not any(
                getattr(item, "role", None) in {"user", "system", "developer"} for item in root.conversation
            ):
                root.conversation = [NeMoGymEasyInputMessage(role="user", content=prompt), *root.conversation]

        if not self.config.track_token_ids and not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("OpenCode produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if trajectory is not None:
            response = response.model_copy(update={_INTERNAL_TRAJECTORY_KEY: trajectory.model_dump(mode="json")})
        return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        episode = await self._create_episode(
            body,
            rollout_id=rollout_id,
            collect_observations=isinstance(rollout_id, str),
        )
        if not isinstance(rollout_id, str):
            return episode.response
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_trajectory = agent_resp_json.pop(_INTERNAL_TRAJECTORY_KEY, None)
            trajectory = (
                scope_opencode_trajectory(TrajectoryRecord.model_validate(raw_trajectory), body, rollout_id)
                if isinstance(raw_trajectory, dict) and rollout_id is not None
                else None
            )
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
                | ({"ng_trajectory": trajectory.model_dump(mode="json")} if trajectory is not None else {})
            )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
