# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer, AgenticMMEVerifyRequest


def response(text: str = "", output: list | None = None) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": output
            if output is not None
            else [
                {
                    "id": "answer",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


@pytest.fixture
def server() -> AgenticMMEServer:
    return AgenticMMEServer(
        config=AgenticMMEConfig(name="agentic_mme", host="localhost", port=1, entrypoint=""),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "golden", "reward"),
    [
        ("<answer>blue</answer>", {"value": "blue", "match_type": "exact"}, 1),
        ("Blue", {"value": "blue", "match_type": "exact"}, 0),
        ("The color is BLUE.", {"value": "blue"}, 1),
        ("2017", {"value": "1"}, 0),
        ("Answer is 1.", {"value": 1}, 1),
        ("0", {"value": 0}, 1),
        ("4.01", {"value": 4, "match_type": "numeric", "tolerance": 0.02}, 1),
        ("4.1", {"value": 4, "match_type": "numeric"}, 0),
        ("four", {"value": 4, "match_type": "numeric"}, 0),
        ("NaN", {"value": 4, "match_type": "numeric"}, 0),
        ("", {"value": "blue"}, 0),
        ("<answer></answer>", {"value": "blue"}, 0),
        ("<think><answer>blue</answer></think>red", {"value": "blue"}, 0),
        ("<thinking>blue</thinking><answer>red</answer>", {"value": "blue"}, 0),
        ("<think>blue", {"value": "blue"}, 0),
        ("blue</think>red", {"value": "blue"}, 0),
        ("<answer>blue", {"value": "blue"}, 1),
        ("None", {"value": None}, 0),
        ("anything", {"value": ""}, 0),
        ("anything", {"value": " "}, 0),
        ("True", {"value": True}, 0),
        ("inf", {"value": float("inf")}, 0),
        ("blue", {"value": ["blue"]}, 0),
        ("blue", {"value": "blue", "match_type": "unsupported"}, 0),
        ("blue", None, 0),
    ],
)
async def test_grading(server, text, golden, reward) -> None:
    body = AgenticMMEVerifyRequest(
        responses_create_params={"input": "question"},
        response=response(text),
        verifier_metadata={"golden_answer": golden},
    )
    result = await server.verify(body)
    assert result.reward == reward
    assert result.evaluation_track == "task_only"
    assert bool(result.failure_reason) == (reward == 0)


@pytest.mark.asyncio
async def test_terminal_tool_and_incomplete_do_not_score(server) -> None:
    call = {"type": "function_call", "name": "crop", "call_id": "a", "arguments": "{}"}
    output = response("blue").model_dump()["output"] + [call]
    body = AgenticMMEVerifyRequest(
        responses_create_params={"input": "q"},
        response=response(output=output),
        verifier_metadata={"golden_answer": {"value": "blue"}},
    )
    assert (await server.verify(body)).reward == 0
    body.response = NeMoGymResponse.model_validate(
        response("blue").model_dump() | {"incomplete_details": {"reason": "max_output_tokens"}}
    )
    assert (await server.verify(body)).failure_reason == "incomplete_response"
