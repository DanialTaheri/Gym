from unittest.mock import MagicMock

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.spatialclaw_pivot.action_utils import (
    canonicalize_spatialclaw_action,
)
from resources_servers.spatialclaw_pivot.app import (
    SpatialClawPivotResourcesServer,
    SpatialClawPivotResourcesServerConfig,
    SpatialClawPivotVerifyRequest,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[NeMoGymResponseOutputText(text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _server() -> SpatialClawPivotResourcesServer:
    return SpatialClawPivotResourcesServer(
        config=SpatialClawPivotResourcesServerConfig(
            host="127.0.0.1", port=8080, entrypoint="", name="pivot"
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_canonicalize_python_calls_ignores_formatting():
    left = canonicalize_spatialclaw_action("x = vlm.ask(InputImages[0], question='what?')")
    right = canonicalize_spatialclaw_action(
        "```python\nx=vlm.ask(InputImages[0],question = 'what?')\n```"
    )
    assert left == right


def test_canonicalize_accepts_terminal_answer_after_preparation():
    action = canonicalize_spatialclaw_action(
        "x = vlm.ask(InputImages[0], question='what?')\nReturnAnswer('B')"
    )
    assert action == {
        "type": "final_answer",
        "answer": "B",
        "scoring_mode": "auto",
    }


def test_canonicalize_rejects_action_after_terminal_answer():
    action = canonicalize_spatialclaw_action(
        "ReturnAnswer('B')\nshow(InputImages[0])"
    )
    assert action is None


def test_canonicalize_preserves_fenced_no_call_program():
    action = canonicalize_spatialclaw_action(
        "```python\n# Preserve an expert no-op decision\nx = InputImages[0]\n```"
    )
    assert action == {"type": "python_calls", "calls": []}
    assert canonicalize_spatialclaw_action("plain_identifier") is None


def test_canonicalize_ignores_calls_in_unexecuted_definitions():
    action = canonicalize_spatialclaw_action(
        "def unused():\n    return sam3.segment(InputImages[0], 'dog')\n"
        "x = vlm.ask(InputImages[0], question='what?')"
    )
    assert action is not None
    assert [call["name"] for call in action["calls"]] == ["vlm.ask"]


async def test_verify_python_call_structure():
    expected = canonicalize_spatialclaw_action("x = sam3.segment(InputImages[0], 'dog')")
    request = SpatialClawPivotVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("```python\nx=sam3.segment(InputImages[0], 'dog')\n```"),
        expected_action=expected,
        pivot_id="trajectory:1",
    )
    result = await _server().verify(request)
    assert result.reward == 1.0
    assert result.category == "matching_python_call_names"


async def test_verify_python_call_names_allow_functional_argument_variants():
    expected = canonicalize_spatialclaw_action(
        "x = sam3.segment(InputImages[0], 'dog')"
    )
    request = SpatialClawPivotVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("x = sam3.segment(InputImages[4], 'animal')"),
        expected_action=expected,
        pivot_id="trajectory:1",
    )
    result = await _server().verify(request)
    assert result.reward == 1.0
    assert result.category == "matching_python_call_names"


async def test_verify_final_answer_mcqa():
    request = SpatialClawPivotVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response("ReturnAnswer('B')"),
        expected_action={"type": "final_answer", "answer": "B", "scoring_mode": "mcqa"},
        pivot_id="trajectory:2",
    )
    result = await _server().verify(request)
    assert result.reward == 1.0
    assert result.category == "matching_final_answer"
