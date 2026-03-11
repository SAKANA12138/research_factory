"""
OpenReview paper search (using API v2).
Correct endpoint: https://api2.openreview.net/notes
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("rf.tools")


async def search_openreview(
    query: str,
    venue: str = "ICLR.cc/2026/Conference",
    limit: int = 30,
) -> list[dict]:
    """Search OpenReview for papers matching the query in a given venue."""
    # API v2: GET /notes with invitation filter
    url = "https://api2.openreview.net/notes"
    # The invitation format is typically: {venue}/-/Submission
    invitation = f"{venue}/-/Submission"
    params = {
        "invitation": invitation,
        "limit": limit,
        "details": "original",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(
                    "OpenReview returned status %d for venue %s", resp.status_code, venue
                )
                return []
            data = resp.json()
    except Exception as e:
        logger.warning("OpenReview search error: %s", e)
        return []

    results = []
    query_lower = query.lower()
    for note in data.get("notes", []):
        content = note.get("content", {})
        title_val = content.get("title", {})
        abstract_val = content.get("abstract", {})
        # API v2: content fields are {value: "..."} dicts
        title = title_val.get("value", "") if isinstance(title_val, dict) else str(title_val)
        abstract = abstract_val.get("value", "") if isinstance(abstract_val, dict) else str(abstract_val)

        # Basic keyword filtering since the API may not support free-text search
        if query_lower and not any(
            kw in (title + " " + abstract).lower()
            for kw in query_lower.split()[:3]  # match on first 3 keywords
        ):
            continue

        results.append(
            {
                "id": note.get("id", ""),
                "title": title,
                "abstract": abstract[:400],
                "venue": venue,
                "url": f"https://openreview.net/forum?id={note.get('id', '')}",
            }
        )
    return results
