# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-field handling shared by CLI harness agents."""

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


# Sampling controls that RL trainers (e.g. NeMo RL's rollout manager) set on every row. A CLI harness
# issues its own model calls, so these never reach the policy through it; the model server's
# sampling_overrides enforce sampling instead.
ROW_SAMPLING_FIELDS = ("temperature", "top_p", "max_output_tokens")


def drop_row_sampling_fields(
    body: NeMoGymResponseCreateParamsNonStreaming,
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Return a copy of ``body`` with the row-level sampling fields reset to their defaults."""
    fields = type(body).model_fields
    return body.model_copy(
        update={name: fields[name].get_default(call_default_factory=True) for name in ROW_SAMPLING_FIELDS}
    )
