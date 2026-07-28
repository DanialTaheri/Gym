from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.spatialclaw.app import (
    SpatialClawResourcesServer,
    SpatialClawResourcesServerConfig,
    SpatialClawVerifyRequest,
    _visible_answer,
)


def _response(answer: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"spatialclaw_final_answer": answer},
    )


def _server() -> SpatialClawResourcesServer:
    return SpatialClawResourcesServer(
        config=SpatialClawResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def test_verify_mcqa_uses_parsed_spatialclaw_answer():
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("ReturnAnswer('C')"),
        expected_answer="C",
        scoring_mode="auto",
    )

    result = await _server().verify(request)

    assert result.reward == 1.0
    assert result.extracted_answer == "ReturnAnswer('C')"
    assert result.scoring_mode_used == "mcqa"


async def test_verify_exact_rejects_wrong_answer():
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("no"),
        expected_answer="yes",
        scoring_mode="exact",
    )

    result = await _server().verify(request)

    assert result.reward == 0.0
    assert result.scorer_supported is True


async def test_verify_token_f1_gives_partial_credit_and_ignores_thinking():
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("A black dog jumps over the wooden fence."),
        expected_answer=(
            "<think>private reference reasoning</think> "
            "The black dog jumps over a fence."
        ),
        scoring_mode="token_f1",
    )

    result = await _server().verify(request)

    assert 0.7 < result.reward < 1.0
    assert result.scoring_mode_used == "token_f1"


def test_visible_answer_removes_malformed_thinking_boundaries():
    assert _visible_answer("reasoning</think>Visible answer") == "Visible answer"
    assert _visible_answer("Visible answer<think>reasoning") == "Visible answer"
    assert (
        _visible_answer("<think>outer <think>nested</think></think>Visible answer")
        == "Visible answer"
    )


async def test_verify_ignores_orphaned_private_reasoning_in_prediction():
    request = SpatialClawVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("private unrelated reasoning</think>The black dog jumps."),
        expected_answer="The black dog jumps.",
        scoring_mode="token_f1",
    )

    result = await _server().verify(request)

    assert result.reward == 1.0
