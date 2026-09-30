# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
from pathlib import Path

from PIL import Image

from responses_api_agents.agentic_mme_agent.mcp_server import ToolServer
from responses_api_agents.agentic_mme_agent.tools import IMAGE_TOOLS


def server(tmp_path: Path, max_tool_calls: int = 15) -> ToolServer:
    Image.new("RGB", (8, 4), "red").save(tmp_path / "image_0.png")
    Image.new("RGB", (4, 4), "blue").save(tmp_path / "image_1.jpg")
    return ToolServer(tmp_path, max_tool_calls)


def rpc(tool_server: ToolServer, method: str, params: dict | None = None, request_id: int | None = 1) -> dict | None:
    message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        message["id"] = request_id
    return tool_server.handle(message)


def test_handshake_and_tool_list(tmp_path: Path) -> None:
    tool_server = server(tmp_path)
    assert rpc(tool_server, "initialize")["result"]["capabilities"] == {"tools": {}}
    assert rpc(tool_server, "notifications/initialized", request_id=None) is None
    tools = rpc(tool_server, "tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == list(IMAGE_TOOLS)
    assert "bbox_2d" in tools[0]["inputSchema"]["properties"]
    assert "error" in rpc(tool_server, "resources/list")


def test_crop_returns_indexed_image(tmp_path: Path) -> None:
    tool_server = server(tmp_path)
    result = rpc(tool_server, "tools/call", {"name": "crop", "arguments": {"image_index": 1, "bbox_2d": [0, 0, 500, 1000]}})
    text, image = result["result"]["content"]
    assert result["result"]["isError"] is False
    assert json.loads(text["text"])["new_image_index"] == 2  # two originals, then the crop
    decoded = Image.open(io.BytesIO(base64.b64decode(image["data"])))
    assert image["mimeType"] == "image/png" and decoded.size == (2, 4)


def test_errors_and_budget(tmp_path: Path) -> None:
    tool_server = server(tmp_path, max_tool_calls=2)
    bad = rpc(tool_server, "tools/call", {"name": "crop", "arguments": {"image_index": "0"}})["result"]
    assert bad["isError"] and not json.loads(bad["content"][0]["text"])["ok"]
    assert not rpc(tool_server, "tools/call", {"name": "flip", "arguments": {"image_index": 0}})["result"]["isError"]
    spent = rpc(tool_server, "tools/call", {"name": "flip", "arguments": {"image_index": 0}})["result"]
    assert spent["isError"] and "budget" in spent["content"][0]["text"]
