# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retrieval through Gym's shared aiohttp transport; explicit Lens upload opt-in.

The web-search provider is configurable (search_provider):
    serper   google.serper.dev            google_search + google_lens_search (Google results)
    serpapi  serpapi.com (engine=google / google_lens), answers mapped to Serper's shape
    tavily   api.tavily.com/search        google_search only (no reverse image search)
    exa      api.exa.ai/search            google_search only (no reverse image search)
fetch_webpage always reads pages through Jina Reader. Every provider's search answer is reduced
to the same Serper-shaped object ({"organic": [{title, link, snippet}], ...}) so the model sees
one format whatever the backend.
"""

import asyncio
import ipaddress
import json
from typing import Any, Literal
from urllib.parse import urlsplit

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


SEARCH_PROVIDERS = ("serper", "serpapi", "tavily", "exa")
# Providers that can answer google_lens_search (reverse image search through Google Lens).
LENS_PROVIDERS = ("serper", "serpapi")


class RetrievalConfig(BaseModel):
    mode: Literal["disabled", "live", "replay"] = "disabled"
    search_provider: Literal["serper", "serpapi", "tavily", "exa"] = "serper"
    serper_api_key: SecretStr = SecretStr("")
    serpapi_api_key: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")
    exa_api_key: SecretStr = SecretStr("")
    imgbb_api_key: SecretStr = SecretStr("")
    jina_api_key: SecretStr = SecretStr("")
    allow_image_upload: bool = False
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_response_bytes: int = Field(default=2_000_000, gt=0, le=10_000_000)

    @model_validator(mode="after")
    def validate_live(self) -> "RetrievalConfig":
        if self.mode == "live" and not self.provider_key():
            raise ValueError(
                f"live retrieval with search_provider={self.search_provider} requires {self.search_provider}_api_key"
            )
        if self.allow_image_upload and not self.imgbb_api_key.get_secret_value():
            raise ValueError("Lens image upload requires imgbb_api_key")
        return self

    def provider_key(self) -> str:
        return getattr(self, f"{self.search_provider}_api_key").get_secret_value()

    def lens_available(self) -> bool:
        return self.search_provider in LENS_PROVIDERS


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


# Serper accepts a bare "zh"; SerpAPI rejects it (HTTP 400 "Unsupported `zh` interface language").
_SERPAPI_HL = {"zh": "zh-cn", "zh-hans": "zh-cn", "zh-sg": "zh-cn", "zh-hant": "zh-tw", "zh-hk": "zh-tw"}


def serpapi_to_serper(data: dict[str, Any], lens: bool) -> dict[str, Any]:
    """SerpAPI google / google_lens answer -> Serper's shape (organic first, as Serper returns)."""
    if lens:
        return {
            "organic": [
                {
                    "title": m.get("title"),
                    "link": m.get("link"),
                    "source": m.get("source"),
                    "imageUrl": m.get("image"),
                    "thumbnailUrl": m.get("thumbnail"),
                }
                for m in data.get("visual_matches") or []
                if isinstance(m, dict)
            ][:10]
        }
    out: dict[str, Any] = {
        "organic": [
            {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet"), "date": r.get("date")}
            for r in data.get("organic_results") or []
            if isinstance(r, dict)
        ][:10]
    }
    kg = data.get("knowledge_graph")
    if isinstance(kg, dict) and kg.get("title"):
        out["knowledgeGraph"] = {k: kg.get(k) for k in ("title", "type", "description", "website")}
    box = data.get("answer_box")
    if isinstance(box, dict):
        out["answerBox"] = {k: box.get(k) for k in ("title", "answer", "snippet", "link")}
    return out


def snippets_to_serper(results: list[Any], text_key: str, link_key: str = "url") -> dict[str, Any]:
    """Tavily / Exa result list -> Serper's organic list (snippet capped at 600 characters)."""
    organic = []
    for r in results or []:
        if isinstance(r, dict):
            text = r.get(text_key)
            if isinstance(text, list):  # Exa highlights
                text = " ... ".join(str(t) for t in text)
            organic.append(
                {
                    "title": r.get("title"),
                    "link": r.get(link_key),
                    "snippet": (str(text or "")[:600]),
                    "date": r.get("publishedDate") or r.get("published_date"),
                }
            )
    return {"organic": organic}


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
        provider, key = self.config.search_provider, self.config.provider_key()
        if isinstance(args, SearchArgs) and provider in ("tavily", "exa"):
            if provider == "tavily":
                raw = await http_payload(
                    self.config,
                    "POST",
                    "https://api.tavily.com/search",
                    headers={"Authorization": "Bearer " + key},
                    json={"query": args.query, "max_results": 5, "search_depth": "basic"},
                )
                data = snippets_to_serper(json.loads(raw).get("results"), "content")
            else:
                raw = await http_payload(
                    self.config,
                    "POST",
                    "https://api.exa.ai/search",
                    headers={"x-api-key": key},
                    json={"query": args.query, "numResults": 5, "contents": {"highlights": True}},
                )
                data = snippets_to_serper(json.loads(raw).get("results"), "highlights")
            return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}
        if isinstance(args, LensArgs) and not self.config.lens_available():
            raise ValueError(f"reverse image search is not available with search_provider={provider}")
        headers = {"X-API-KEY": key}
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
        if provider == "serpapi":
            params = {"api_key": key, "num": "5"}
            if endpoint == "search":
                params |= {
                    "engine": "google",
                    "q": args.query,
                    "gl": args.gl,
                    "hl": _SERPAPI_HL.get(args.hl.lower(), args.hl),
                }
            else:
                params |= {"engine": "google_lens", "url": payload["url"]}
            raw = await http_payload(self.config, "GET", "https://serpapi.com/search.json", params=params)
            data = serpapi_to_serper(json.loads(raw), lens=endpoint == "lens")
        else:
            raw = await http_payload(
                self.config, "POST", "https://google.serper.dev/" + endpoint, headers=headers, json=payload
            )
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval provider returned an invalid JSON object")
        return {"ok": True, "tool": name, "context": json.dumps(data, ensure_ascii=False)[:50000], "raw": data}
