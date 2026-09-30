# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-MME atomic image tools as a stdio MCP server, for CLI harnesses such as OpenCode.

    python mcp_server.py --images-dir <dir with image_0.png, image_1.jpg, ...> [--max-tool-calls 15]

Same tools, schemas and semantics as agentic_mme_agent (tools.ImageWorkspace): the task's
images are indexed 0, 1, ... in file order, each successful operation appends one image, and
every attempted call, valid or not, consumes the budget. A result carries the new image as MCP
`image` content so the harness shows it to the model, like agentic_mme_agent's image message.

Only the protocol surface a harness uses is implemented (initialize, tools/list, tools/call,
ping) as newline-delimited JSON-RPC 2.0 on stdin/stdout.
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responses_api_agents.agentic_mme_agent.tools import IMAGE_TOOLS, ImageWorkspace  # noqa: E402


PROTOCOL_VERSION = "2024-11-05"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class ToolServer:
    def __init__(self, images_dir: Path, max_tool_calls: int) -> None:
        self.workspace = ImageWorkspace()
        for path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            self.workspace.load(image_data_url_from_file(path))
        self.max_tool_calls = max_tool_calls
        self.attempts = 0

    def tools(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": description, "inputSchema": model.model_json_schema()}
            for name, (model, description) in IMAGE_TOOLS.items()
        ]

    def call(self, name: str, arguments: Any) -> tuple[list[dict[str, Any]], bool]:
        try:
            if self.attempts >= self.max_tool_calls:
                raise ValueError("tool budget exhausted; provide a final answer")
            self.attempts += 1  # Invalid calls consume budget too.
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            result = self.workspace.apply(name, arguments)
        except Exception as exc:  # Every failure goes back to the model as a tool error.
            return [{"type": "text", "text": json.dumps({"ok": False, "error": str(exc)})}], True
        header, data = result.pop("image_url").split(",", 1)
        mime = header.removeprefix("data:").split(";", 1)[0]
        return [
            {"type": "text", "text": json.dumps(result)},
            {"type": "image", "data": data, "mimeType": mime},
        ], False

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method, request_id = message.get("method"), message.get("id")
        if request_id is None:  # Notifications (e.g. notifications/initialized) get no reply.
            return None
        params = message.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agentic-mme", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            content, is_error = self.call(params.get("name", ""), params.get("arguments") or {})
            result = {"content": content, "isError": is_error}
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"unknown {method}"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def image_data_url_from_file(path: Path) -> str:
    """The file's own bytes, as the harness received them; no re-encoding."""
    mime = {".jpg": "jpeg"}.get(path.suffix.lower(), path.suffix.lower()[1:])
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    args = parser.parse_args()
    server = ToolServer(args.images_dir, args.max_tool_calls)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = server.handle(json.loads(line))
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
