# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task images for CLI harnesses: read them from a Responses input, stage them as files.

CLI harnesses (Codex, Pi) take images as files on their command line, not as message parts. These
helpers pull the data-URL images out of the task's last user message and write them where the
harness runs, so the CLI attaches them to its first model call and its tools can open them.
"""

import base64
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox


IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}
_DATA_URL = re.compile(r"data:(image/[a-z]+);base64,(.*)", re.S)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def part_type(part: Any) -> Any:
    """The ``type`` of a message content part, a dict or a model."""
    return _field(part, "type")


def extract_input_images(body_input: Iterable[Any] | str) -> list[tuple[str, bytes]]:
    """(suffix, bytes) of the supported data-URL images in the last user message, in order.

    Remote URLs and unsupported image types are skipped: the harness runs offline and only these
    formats are decoded by the CLIs and the model server alike.
    """
    if isinstance(body_input, str):
        return []
    for item in reversed(list(body_input)):
        if _field(item, "role") != "user":
            continue
        content = _field(item, "content")
        images = []
        for part in content if isinstance(content, list) else []:
            if part_type(part) != "input_image":
                continue
            match = _DATA_URL.match(_field(part, "image_url") or "")
            if match and match.group(1) in IMAGE_SUFFIXES:
                images.append((IMAGE_SUFFIXES[match.group(1)], base64.b64decode(match.group(2))))
        return images
    return []


def image_file_names(images: list[tuple[str, bytes]]) -> list[str]:
    """``image_<i>.<ext>``, the names OpenCode uses, so prompts read the same across harnesses."""
    return [f"image_{index}{suffix}" for index, (suffix, _) in enumerate(images)]


async def upload_bytes(sandbox: AsyncSandbox, *, path: str, data: bytes) -> None:
    """Upload raw bytes through file transfer, without shell interpolation."""
    with tempfile.TemporaryDirectory(prefix="sandbox-upload-") as directory:
        source = Path(directory) / "payload"
        source.write_bytes(data)
        await sandbox.upload(source, path)
