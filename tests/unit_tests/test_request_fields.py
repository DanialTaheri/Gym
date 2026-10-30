# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nemo_gym.agent_utils.request_fields import ROW_SAMPLING_FIELDS, drop_row_sampling_fields
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def test_drop_row_sampling_fields_resets_only_the_sampling_fields() -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="What color is the circle?",
        instructions="Answer briefly.",
        temperature=1.0,
        top_p=0.9,
        max_output_tokens=98304,
    )

    dropped = drop_row_sampling_fields(body)

    defaults = NeMoGymResponseCreateParamsNonStreaming(input="x")
    for name in ROW_SAMPLING_FIELDS:
        assert getattr(dropped, name) == getattr(defaults, name)
    assert dropped.input == "What color is the circle?"
    assert dropped.instructions == "Answer briefly."
    # The caller's request is left as it was.
    assert (body.temperature, body.top_p, body.max_output_tokens) == (1.0, 0.9, 98304)
