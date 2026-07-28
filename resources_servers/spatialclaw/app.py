# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reward server for parsed SpatialClaw ReturnAnswer values."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class SpatialClawResourcesServerConfig(BaseResourcesServerConfig):
    spatialclaw_root: str = ""


class SpatialClawVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    expected_answer: str = ""
    benchmark: str | None = None
    sample_id: str | int | None = None
    dataset_config: str | None = None
    data_root: str | None = None
    scoring_mode: Literal["auto", "mcqa", "exact", "token_f1", "native"] = "auto"


class SpatialClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    expected_answer: str
    extracted_answer: str
    scoring_mode_used: str
    scorer_supported: bool = True


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _visible_answer(value: Any) -> str:
    """Remove balanced or malformed private thinking spans from an answer."""
    text = str(value or "")
    visible: list[str] = []
    hidden_depth = 0
    cursor = 0
    for match in re.finditer(r"(?is)</?think>", text):
        if hidden_depth == 0:
            visible.append(text[cursor : match.start()])
        if match.group().casefold() == "<think>":
            hidden_depth += 1
        elif hidden_depth:
            hidden_depth -= 1
        else:
            # A dangling closing tag means the preceding text was an
            # untagged private reasoning preamble.
            visible.clear()
        cursor = match.end()
    if hidden_depth == 0:
        visible.append(text[cursor:])
    return " ".join(part.strip() for part in visible if part.strip()).strip()


def _extract_choice(value: Any) -> str:
    text = _normalize(_visible_answer(value))
    patterns = (
        r"^([A-Za-z])$",
        r"\\boxed\{\s*([A-Za-z])\s*\}",
        r"(?i)(?:answer|choice|option)\s*(?:is|:)?\s*([A-Za-z])\b",
        r"ReturnAnswer\(\s*['\"]([A-Za-z])['\"]\s*\)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).upper()
    return text.upper()


def _answer_tokens(value: Any) -> list[str]:
    """SQuAD-style normalized word tokens for deterministic partial credit."""
    text = _normalize(_visible_answer(value)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    return [token for token in tokens if token not in {"a", "an", "the"}]


def _token_f1(prediction: Any, expected: Any) -> float:
    prediction_tokens = _answer_tokens(prediction)
    expected_tokens = _answer_tokens(expected)
    if not prediction_tokens or not expected_tokens:
        return float(prediction_tokens == expected_tokens and bool(expected_tokens))
    overlap = sum((Counter(prediction_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


class SpatialClawResourcesServer(SimpleResourcesServer):
    config: SpatialClawResourcesServerConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _benchmark_cache: ClassVar[dict[str, Any]] = {}

    def _root(self) -> Path:
        root = Path(self.config.spatialclaw_root).expanduser().resolve()
        if not (root / "spatial_agent").is_dir():
            raise RuntimeError(f"Invalid SpatialClaw checkout for verifier: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    def _native_score(self, body: SpatialClawVerifyRequest, prediction: str) -> float | None:
        if not body.benchmark or body.sample_id is None:
            return None
        root = self._root()
        key = json.dumps(
            {
                "benchmark": body.benchmark,
                "dataset_config": body.dataset_config,
                "data_root": body.data_root,
            },
            sort_keys=True,
        )
        benchmark = self._benchmark_cache.get(key)
        if benchmark is None:
            from spatial_agent.config import SpatialAgentConfig, set_config
            from spatial_agent.evals.factory import BenchmarkFactory

            config = SpatialAgentConfig()
            if body.dataset_config:
                path = Path(body.dataset_config)
                if not path.is_absolute():
                    path = root / "spatial_agent" / "config" / "dataset" / body.dataset_config
                if path.suffix != ".json":
                    path = path.with_suffix(".json")
                config.update_from_dataset_json(str(path))
            set_config(config)
            benchmark = BenchmarkFactory.create_benchmark(
                body.benchmark,
                data_root=body.data_root or str(root / "data"),
                question_type=getattr(config, "question_type", None),
            )
            self._benchmark_cache[key] = benchmark
        sample = next(
            (sample for sample in benchmark if str(sample.sample_id) == str(body.sample_id)),
            None,
        )
        if sample is None:
            raise KeyError(
                f"SpatialClaw sample {body.sample_id!r} not found in benchmark {body.benchmark!r}"
            )
        return benchmark.evaluate_single(sample, prediction)

    async def verify(self, body: SpatialClawVerifyRequest) -> SpatialClawVerifyResponse:
        metadata = body.response.metadata or {}
        prediction = str(metadata.get("spatialclaw_final_answer", ""))
        expected = _normalize(body.expected_answer)
        mode = body.scoring_mode
        score: float | None = None
        mode_used = mode

        if mode in {"native", "auto"} and body.benchmark and body.sample_id is not None:
            score = self._native_score(body, prediction)
            mode_used = "native"
        if score is None and mode in {"mcqa", "auto"} and len(_extract_choice(expected)) == 1:
            score = float(bool(prediction) and _extract_choice(prediction) == _extract_choice(expected))
            mode_used = "mcqa"
        if score is None and mode in {"exact", "auto"}:
            score = float(bool(prediction) and _normalize(prediction).casefold() == expected.casefold())
            mode_used = "exact"
        if score is None and mode == "token_f1":
            score = _token_f1(prediction, expected) if prediction else 0.0
            mode_used = "token_f1"

        supported = score is not None
        return SpatialClawVerifyResponse(
            **body.model_dump(exclude={"expected_answer"}),
            reward=float(score or 0.0),
            expected_answer=expected,
            extracted_answer=_normalize(prediction),
            scoring_mode_used=mode_used,
            scorer_supported=supported,
        )


if __name__ == "__main__":
    SpatialClawResourcesServer.run_webserver()
