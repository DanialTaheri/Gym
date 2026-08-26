# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-step functional verifier for SpatialClaw PivotRL actions."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_tool_call_or_text,
)
from resources_servers.spatialclaw.app import _extract_choice, _normalize, _token_f1
from resources_servers.spatialclaw_pivot.action_utils import (
    canonicalize_spatialclaw_action,
)


class SpatialClawPivotExpectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["python_calls", "final_answer"]
    calls: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    scoring_mode: Literal["auto", "mcqa", "exact", "token_f1"] = "auto"


class SpatialClawPivotResourcesServerConfig(BaseResourcesServerConfig):
    python_call_comparison: Literal["names", "canonical"] = "names"


class SpatialClawPivotRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    expected_action: SpatialClawPivotExpectedAction
    pivot_id: str = ""
    trajectory_id: str = ""
    decision_index: int = 0
    session_kind: str = "main"


class SpatialClawPivotVerifyRequest(SpatialClawPivotRunRequest, BaseVerifyRequest):
    pass


class SpatialClawPivotVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    expected_action: SpatialClawPivotExpectedAction
    actual_action: dict[str, Any] | None = None
    pivot_id: str = ""
    category: str


def _score_final_answer(prediction: str, expected: str, mode: str) -> float:
    if mode in {"mcqa", "auto"} and len(_extract_choice(expected)) == 1:
        return float(
            bool(prediction)
            and _extract_choice(prediction) == _extract_choice(expected)
        )
    if mode in {"exact", "auto"}:
        return float(
            bool(prediction)
            and _normalize(prediction).casefold() == _normalize(expected).casefold()
        )
    if mode == "token_f1":
        return _token_f1(prediction, expected) if prediction else 0.0
    return 0.0


class SpatialClawPivotResourcesServer(SimpleResourcesServer):
    config: SpatialClawPivotResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(
        self, body: SpatialClawPivotVerifyRequest
    ) -> SpatialClawPivotVerifyResponse:
        extracted = extract_tool_call_or_text(body.response)
        text = extracted.text if extracted is not None and extracted.type == "output_text" else ""
        actual = canonicalize_spatialclaw_action(text)
        expected = body.expected_action

        if actual is None:
            reward, category = 0.0, "no_spatialclaw_action"
        elif expected.type == "python_calls":
            if actual.get("type") != "python_calls":
                reward, category = 0.0, "expected_python_calls"
            elif self.config.python_call_comparison == "canonical":
                reward = float(actual.get("calls") == expected.calls)
                category = (
                    "matching_python_calls"
                    if reward
                    else "different_python_calls"
                )
            else:
                expected_names = [call.get("name") for call in expected.calls]
                actual_names = [call.get("name") for call in actual.get("calls", [])]
                reward = float(actual_names == expected_names)
                category = (
                    "matching_python_call_names"
                    if reward
                    else "different_python_call_names"
                )
        elif actual.get("type") != "final_answer":
            reward, category = 0.0, "expected_final_answer"
        else:
            reward = _score_final_answer(
                str(actual.get("answer", "")), expected.answer, expected.scoring_mode
            )
            category = "matching_final_answer" if reward else "different_final_answer"

        return SpatialClawPivotVerifyResponse(
            **body.model_dump(),
            reward=reward,
            actual_action=actual,
            category=category,
        )


if __name__ == "__main__":
    SpatialClawPivotResourcesServer.run_webserver()
