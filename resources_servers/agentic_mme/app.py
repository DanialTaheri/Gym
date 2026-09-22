# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic-MME task-only scoring; process judgments are deliberately not rewards."""

import math
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class GoldenAnswer(BaseModel):
    value: str | int | float
    match_type: Literal["exact", "contains", "numeric"] = "contains"
    tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("value", mode="before")
    @classmethod
    def valid_target(cls, value: Any) -> str | int | float:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
            raise ValueError("golden_answer.value must be a nonempty string or finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("golden_answer.value must be finite")
        return value


def extract_answer(text: str) -> str:
    """Extract the public atomic answer format, excluding private reasoning."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.S | re.I)
    text = re.split(r"</(?:think|thinking)>", text, flags=re.I)[-1]
    text = re.split(r"<(?:think|thinking)>", text, flags=re.I)[0]
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S | re.I)
    return (match.group(1) if match else text).strip()


def matches_answer(answer: str, golden: GoldenAnswer) -> bool:
    """Match the released evaluator's exact/contains/numeric task semantics.

    The paper says normalized exact match, but the released dataset/evaluator
    default to contains. Do not silently substitute Gym's soft numeric grader.
    """
    target = str(golden.value).strip()
    if not answer:
        return False
    if golden.match_type == "exact":
        return answer == target
    if golden.match_type == "numeric":
        try:
            actual, expected = float(answer), float(target)
        except ValueError:
            return False
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= golden.tolerance
    numeric = target.replace(".", "").replace("-", "").replace("+", "").isdigit()
    if numeric and len(target) <= 2:
        return re.search(r"\b" + re.escape(target) + r"\b", answer) is not None
    return target.lower() in answer.lower()


class AgenticMMEConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class AgenticMMEVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticMMEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    extracted_answer: str = ""
    failure_reason: str | None = None
    evaluation_track: str = "task_only"


class AgenticMMEServer(SimpleResourcesServer):
    config: AgenticMMEConfig

    async def verify(self, body: AgenticMMEVerifyRequest) -> AgenticMMEVerifyResponse:
        result = AgenticMMEVerifyResponse(**body.model_dump(), reward=0.0)
        try:
            golden = GoldenAnswer.model_validate(body.verifier_metadata.get("golden_answer"))
        except ValidationError:
            result.failure_reason = "invalid_golden_answer"
            return result
        # Only the terminal turn may answer; never score an earlier tool-turn message.
        terminal = []
        for item in body.response.output:
            if item.type in ("function_call", "function_call_output"):
                terminal = []
            elif item.type == "message" and item.role == "assistant":
                terminal = [part.text for part in item.content if part.type == "output_text"]
        if body.response.incomplete_details:
            result.failure_reason = "incomplete_response"
            return result
        result.extracted_answer = extract_answer("\n".join(terminal))
        result.reward = float(matches_answer(result.extracted_answer, golden))
        if not result.reward:
            result.failure_reason = "incorrect_answer" if result.extracted_answer else "missing_answer"
        return result


if __name__ == "__main__":
    AgenticMMEServer.run_webserver()
