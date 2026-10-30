# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
from pathlib import Path

from nemo_gym.agent_utils.input_images import extract_input_images, image_file_names, upload_bytes
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def test_images_come_from_the_last_user_message_in_order():
    body_input = [
        {"role": "user", "content": [{"type": "input_image", "image_url": _data_url("image/png", b"old")}]},
        {"role": "assistant", "content": "seen"},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": _data_url("image/jpeg", b"first")},
                {"type": "input_text", "text": "question"},
                {"type": "input_image", "image_url": _data_url("image/png", b"second")},
            ],
        },
    ]
    images = extract_input_images(body_input)
    assert images == [(".jpg", b"first"), (".png", b"second")]
    assert image_file_names(images) == ["image_0.jpg", "image_1.png"]


def test_remote_and_unsupported_images_are_skipped():
    body_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/a.png"},
                {"type": "input_image", "image_url": _data_url("image/gif", b"gif")},
                {"type": "input_text", "text": "q"},
            ],
        }
    ]
    assert extract_input_images(body_input) == []
    assert extract_input_images("plain text prompt") == []


def test_model_parts_are_read_like_dicts():
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": _data_url("image/webp", b"w"), "detail": "auto"},
                    {"type": "input_text", "text": "q"},
                ],
            }
        ]
    )
    assert extract_input_images(body.input) == [(".webp", b"w")]


def test_upload_bytes_hands_the_exact_bytes_to_the_sandbox():
    received = {}

    class Sandbox:
        async def upload(self, source, destination):
            received[destination] = Path(source).read_bytes()

    asyncio.run(upload_bytes(Sandbox(), path="/app/image_0.png", data=b"\x89PNG\x00\xff"))
    assert received == {"/app/image_0.png": b"\x89PNG\x00\xff"}
