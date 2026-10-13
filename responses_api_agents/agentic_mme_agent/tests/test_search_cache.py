# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from PIL import Image

from responses_api_agents.agentic_mme_agent import retrieval
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval, RetrievalConfig
from responses_api_agents.agentic_mme_agent.search_cache import SearchCache, normalize_query, shared_cache
from responses_api_agents.agentic_mme_agent.sqlite_fuzzy_index import build_master_db, master_db_path
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace


def _seed(root, query, results):
    cache = SearchCache(str(root), fuzzy_threshold=0)
    cache.put_search(query, results, "serpapi", "us", "en")
    return cache


def test_exact_fuzzy_and_number_guard(tmp_path) -> None:
    _seed(tmp_path, "Eiffel Tower height in meters 2010", {"organic": [{"title": "2010"}]})
    build_master_db(master_db_path(str(tmp_path)), [normalize_query("Eiffel Tower height in meters 2010")])
    cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
    assert cache.get_search("eiffel tower height in meters 2010")[1] == "exact"
    assert cache.get_search("height in meters of the Eiffel Tower 2010")[1] == "fuzzy"
    # same words, different year: not the same question
    assert cache.get_search("Eiffel Tower height in meters 2020") == (None, "miss")


@pytest.mark.asyncio
async def test_miss_calls_provider_then_later_calls_hit_without_network(tmp_path, monkeypatch) -> None:
    shared_cache.cache_clear()
    calls = []

    async def fake_payload(cfg, method, url, **kwargs):
        calls.append(url)
        if "imgbb" in url:
            return json.dumps({"success": True, "data": {"url": "https://i.ibb.co/x.png"}}).encode()
        if "serpapi" in url:
            return json.dumps({"visual_matches": [{"title": "M", "link": "https://m.example"}]}).encode()
        if "jina" in url:
            return b"page text " * 50
        return json.dumps({"results": [{"title": "T", "url": "https://t.example", "content": "c"}]}).encode()

    monkeypatch.setattr(retrieval, "http_payload", fake_payload)
    config = RetrievalConfig(
        mode="live",
        search_provider="tavily",
        tavily_api_key="t",
        lens_provider="serpapi",
        serpapi_api_key="s",
        imgbb_api_key="i",
        allow_image_upload=True,
        cache_root=str(tmp_path),
    )
    workspace = ImageWorkspace()
    workspace.add(Image.new("RGB", (4, 4), "red"))
    first = Retrieval(config)
    assert (await first.call("google_search", {"query": "Q one"}, workspace))["cache"] == "miss"
    assert (await first.call("google_lens_search", {"image_index": 0}, workspace))["cache"] == "miss"
    page = await first.call("fetch_webpage", {"url": "https://a.example/p", "max_chars": 20}, workspace)
    assert page["cache"] == "miss" and "full_text" not in page and len(page["context"]) == 20
    n = len(calls)
    second = Retrieval(config)  # a later rollout in the same process
    assert (await second.call("google_search", {"query": "  q ONE "}, workspace))["cache"] == "exact"
    lens = await second.call("google_lens_search", {"image_index": 0}, workspace)
    assert lens["cache"] == "exact" and lens["raw"]["organic"][0]["title"] == "M"
    page = await second.call("fetch_webpage", {"url": "https://a.example/p/", "max_chars": 100}, workspace)
    assert page["cache"] == "exact" and len(page["context"]) == 100  # the whole page was cached
    assert len(calls) == n  # no provider call for any hit


def test_rebuild_index_makes_new_entries_fuzzy_matchable(tmp_path) -> None:
    from responses_api_agents.agentic_mme_agent.search_cache import rebuild_index

    _seed(tmp_path, "tallest building in Paris 2024", {"organic": []})
    assert SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024")[1] == "miss"
    assert rebuild_index(str(tmp_path)) == 1
    assert SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024")[1] == "fuzzy"
