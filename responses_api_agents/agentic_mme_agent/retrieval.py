# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retrieval through Gym's shared aiohttp transport; explicit Lens upload opt-in.

mode "local" uses an offline Wikipedia retrieval service instead of the internet (no API keys):
google_search -> POST {retriever_url}/search, google_lens_search -> POST {retriever_url}/lens with
the image itself, fetch_webpage -> GET {retriever_url}/page for Wikipedia articles only. The
service (E5 text index + SigLIP image index over Wikipedia) answers in Serper's JSON shape.
"""

import asyncio
import ipaddress
import json
import random
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from nemo_gym.server_utils import request
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace, image_data_url


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    gl: str = Field(default="us", pattern=r"^[a-z]{2}$")
    hl: str = Field(default="en", pattern=r"^[a-z-]{2,10}$")


class LensArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_index: int = Field(default=0, ge=0)


class FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=4096)
    max_chars: int = Field(default=12000, ge=1, le=50000)


RETRIEVAL_TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "google_search": (SearchArgs, "Search Google for external evidence."),
    "google_lens_search": (LensArgs, "Reverse image search on an image_index; defaults to Image 0."),
    "fetch_webpage": (FetchArgs, "Read a public HTTP(S) webpage as text."),
}

# Tool names stay the same in mode "local"; the descriptions say what is really searched.
LOCAL_DESCRIPTIONS: dict[str, str] = {
    "google_search": "Search an offline snapshot of English Wikipedia; returns matching passages with "
    "article titles and links.",
    "google_lens_search": "Reverse image search over Wikipedia's images (offline) on an image_index; "
    "defaults to Image 0. Returns the articles whose images look most similar.",
    "fetch_webpage": "Read a Wikipedia article (offline) as text; other websites are not available.",
}


class RetrievalConfig(BaseModel):
    mode: Literal["disabled", "live", "replay", "local"] = "disabled"
    # mode "local": base URL of the offline Wikipedia retrieval service (wiki_retrieval/server.py);
    # a comma-separated list spreads calls over several service replicas at random.
    retriever_url: str = ""
    serper_api_key: SecretStr = SecretStr("")
    imgbb_api_key: SecretStr = SecretStr("")
    jina_api_key: SecretStr = SecretStr("")
    allow_image_upload: bool = False
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_response_bytes: int = Field(default=2_000_000, gt=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_live(self) -> "RetrievalConfig":
        if self.mode == "live" and not self.serper_api_key.get_secret_value():
            raise ValueError("live retrieval requires serper_api_key")
        if self.mode == "local" and not all(
            u.strip().startswith(("http://", "https://")) for u in self.retriever_url.split(",")
        ):
            raise ValueError("local retrieval requires retriever_url (http://host:port[,http://host2:port])")
        if self.allow_image_upload and not self.imgbb_api_key.get_secret_value():
            raise ValueError("Lens image upload requires imgbb_api_key")
        return self


def public_url(url: str) -> str:
    """Reject local targets; webpage fetching itself is delegated to fixed Jina infrastructure."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise ValueError("a public HTTP(S) URL without credentials is required")
    if parsed.port not in (None, 80, 443):
        raise ValueError("only standard HTTP(S) ports are supported")
    if "." not in host or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("local hostnames are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Numeric host aliases can resolve to loopback despite not being dotted IPv4.
        if host.replace(".", "").isdigit() or host.lower().startswith("0x"):
            raise ValueError("numeric host aliases are not allowed") from None
    else:
        if not address.is_global:
            raise ValueError("non-public addresses are not allowed")
    return url


async def http_payload(config: RetrievalConfig, method: str, url: str, **kwargs: Any) -> bytes:
    # The timeout covers Gym's retry loop as well as reading the response.
    async with asyncio.timeout(config.timeout_seconds):
        response = await request(method, url, allow_redirects=False, **kwargs)
        try:
            if not 200 <= response.status < 300:
                # Do not expose provider URLs, request headers, or secrets in tool errors.
                raise ValueError(f"retrieval provider returned HTTP {response.status}")
            payload = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                payload.extend(chunk)
                if len(payload) > config.max_response_bytes:
                    raise ValueError("retrieval response exceeds the configured byte limit")
            return bytes(payload)
        finally:
            response.release()


class Retrieval:
    def __init__(self, config: RetrievalConfig, replay: list[dict[str, Any]] | None = None) -> None:
        self.config = config
        self.replay = list(replay or [])
        self.replay_position = 0

    async def call(self, name: str, arguments: dict[str, Any], workspace: ImageWorkspace) -> dict[str, Any]:
        if self.config.mode == "disabled":
            raise ValueError("retrieval is disabled for this configuration")
        model = RETRIEVAL_TOOLS[name][0]
        args = model.model_validate(arguments)
        if isinstance(args, FetchArgs):
            public_url(args.url)
        if isinstance(args, LensArgs):
            workspace.get(args.image_index)
        if self.config.mode == "replay":
            if self.replay_position >= len(self.replay):
                raise ValueError("no recorded retrieval result remains")
            event = self.replay[self.replay_position]
            if event["tool_name"] != name or model.model_validate(event["arguments"]) != args:
                raise ValueError("retrieval replay mismatch; no live fallback is permitted")
            self.replay_position += 1
            return event["output"]
        if self.config.mode == "local":
            return await self._local(name, args, workspace)
        if isinstance(args, FetchArgs):
            headers = {"Accept": "text/plain"}
            if self.config.jina_api_key.get_secret_value():
                headers["Authorization"] = "Bearer " + self.config.jina_api_key.get_secret_value()
            raw = await http_payload(self.config, "GET", "https://r.jina.ai/" + args.url, headers=headers)
            return {
                "ok": True,
                "tool": name,
                "url": args.url,
                "context": raw.decode("utf-8", errors="replace")[: args.max_chars],
            }
        headers = {"X-API-KEY": self.config.serper_api_key.get_secret_value()}
        if isinstance(args, SearchArgs):
            payload = {"q": args.query, "gl": args.gl, "hl": args.hl, "num": 5}
            endpoint = "search"
        else:
            if not self.config.allow_image_upload:
                raise ValueError("Lens requires allow_image_upload=true; images are uploaded to ImgBB")
            data_url = image_data_url(workspace.get(args.image_index))
            upload = await http_payload(
                self.config,
                "POST",
                "https://api.imgbb.com/1/upload",
                data={
                    "key": self.config.imgbb_api_key.get_secret_value(),
                    "expiration": "600",
                    "image": data_url.split(",", 1)[1],
                },
            )
            uploaded = json.loads(upload)
            if not uploaded.get("success") or not uploaded.get("data", {}).get("url"):
                raise ValueError("image upload did not return an image URL")
            payload = {"url": public_url(uploaded["data"]["url"]), "num": 5}
            endpoint = "lens"
        raw = await http_payload(
            self.config, "POST", "https://google.serper.dev/" + endpoint, headers=headers, json=payload
        )
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval provider returned an invalid JSON object")
        return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}

    async def _local(self, name: str, args: BaseModel, workspace: ImageWorkspace) -> dict[str, Any]:
        """Offline Wikipedia service; same result shapes as live mode."""
        base = random.choice([u.strip() for u in self.config.retriever_url.split(",")]).rstrip("/")
        if isinstance(args, FetchArgs):
            host = urlsplit(args.url).hostname or ""
            if not (host == "wikipedia.org" or host.endswith(".wikipedia.org")):
                raise ValueError("only Wikipedia articles are available in this offline environment")
            try:
                raw = await http_payload(self.config, "GET", f"{base}/page?url={quote(args.url, safe='')}")
            except ValueError as e:
                if "HTTP 404" in str(e):
                    raise ValueError("no offline Wikipedia article for this URL") from None
                raise
            page = json.loads(raw)
            text = f"Title: {page['title']}\n\nURL Source: {page['url']}\n\nMarkdown Content:\n{page['text']}"
            return {"ok": True, "tool": name, "url": args.url, "context": text[: args.max_chars]}
        if isinstance(args, SearchArgs):
            raw = await http_payload(self.config, "POST", f"{base}/search", json={"query": args.query, "k": 5})
        else:
            data_url = image_data_url(workspace.get(args.image_index))
            raw = await http_payload(
                self.config, "POST", f"{base}/lens", json={"image_b64": data_url.split(",", 1)[1], "k": 5}
            )
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval provider returned an invalid JSON object")
        return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}
