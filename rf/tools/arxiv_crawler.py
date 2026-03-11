"""
arXiv API search wrapper.
The `arxiv` library is synchronous — we wrap it with asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
import logging

import arxiv

logger = logging.getLogger("rf.tools")


def _sync_search_arxiv(query: str, max_results: int, sort_by: arxiv.SortCriterion) -> list[dict]:
    """Synchronous arXiv search (runs in a thread)."""
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
    results = []
    try:
        for paper in search.results():
            results.append(
                {
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "abstract": paper.summary[:500],
                    "arxiv_id": paper.entry_id,
                    "published": paper.published.isoformat() if paper.published else "",
                    "pdf_url": paper.pdf_url or "",
                    "categories": list(paper.categories) if paper.categories else [],
                }
            )
    except Exception as e:
        logger.warning("arXiv search error: %s", e)
    return results


async def search_arxiv(
    query: str,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
) -> list[dict]:
    """Return a list of paper metadata dicts from arXiv (async-safe)."""
    return await asyncio.to_thread(_sync_search_arxiv, query, max_results, sort_by)
