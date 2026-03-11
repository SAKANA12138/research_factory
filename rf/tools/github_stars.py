"""GitHub trending repository tracker using REST API."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("rf.tools")


async def fetch_trending_repos(
    query: str,
    min_stars: int = 100,
    max_results: int = 20,
) -> list[dict]:
    """Search GitHub repos by topic, sorted by stars, return metadata."""
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"{query} stars:>={min_stars}",
        "sort": "stars",
        "order": "desc",
        "per_page": min(max_results, 100),
    }
    headers = {"Accept": "application/vnd.github+json"}
    # Use token if available to avoid rate limits
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("GitHub search error: %s", e)
        return []

    return [
        {
            "full_name": r.get("full_name", ""),
            "stars": r.get("stargazers_count", 0),
            "description": (r.get("description") or "")[:200],
            "url": r.get("html_url", ""),
            "language": r.get("language", ""),
            "updated_at": r.get("updated_at", ""),
            "topics": r.get("topics", []),
        }
        for r in data.get("items", [])
    ]
