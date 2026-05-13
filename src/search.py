"""
search.py — Query processing, suggestions, and result formatting.

Sits on top of the Indexer and adds:
  * Query validation and normalisation
  * Query suggestions (prefix match + Levenshtein via difflib)
  * Formatted result presentation
  * Benchmarking helpers
"""

from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from indexer import tokenise, PageEntry

if TYPE_CHECKING:
    from indexer import Indexer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result returned from a search query."""

    rank: int
    url: str
    title: str
    frequency: int
    tf_idf_score: float
    positions: list[int]


class SearchEngine:
    """
    High-level query interface over an Indexer.

    Parameters
    ----------
    indexer       : A loaded/built Indexer instance.
    max_suggestions : Maximum number of query suggestions to return.
    """

    def __init__(self, indexer: "Indexer", max_suggestions: int = 5) -> None:
        self._indexer = indexer
        self._max_suggestions = max_suggestions

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def find(self, query: str) -> list[SearchResult]:
        """
        Return pages containing ALL terms in query, ranked by TF-IDF.

        Parameters
        ----------
        query : Raw query string (case-insensitive, any whitespace).

        Returns
        -------
        list[SearchResult]
            Ranked results. Empty list for empty/unmatched queries.
        """
        if not query or not query.strip():
            return []

        t0 = time.perf_counter()
        entries = self._indexer.find(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = [
            SearchResult(
                rank=i + 1,
                url=e.url,
                title=self._indexer.page_metadata.get(e.url, {}).get("title", e.url),
                frequency=e.frequency,
                tf_idf_score=e.tf_idf,
                positions=e.positions[:10],
            )
            for i, e in enumerate(entries)
        ]

        logger.debug("find(%r) -> %d results in %.2f ms", query, len(results), elapsed_ms)
        return results

    def print_word(self, word: str) -> list[PageEntry]:
        """
        Return the raw posting list for a single word.

        Parameters
        ----------
        word : Single token to look up.
        """
        if not word or not word.strip():
            return []
        return self._indexer.print_index(word.strip())

    # ------------------------------------------------------------------
    # Query suggestions
    # ------------------------------------------------------------------

    def suggest(self, query: str, n: int | None = None) -> list[str]:
        """
        Return up to n query suggestions for query.

        Tries in order:
        1. Exact match (already valid — no suggestion needed).
        2. Prefix match (query is a prefix of known terms).
        3. Close match via difflib (handles typos).

        Parameters
        ----------
        query : Partial or misspelled query term.
        n     : Max suggestions (defaults to self._max_suggestions).
        """
        n = n or self._max_suggestions
        term = query.strip().lower()
        if not term:
            return []

        vocab = list(self._indexer.index.keys())

        if term in self._indexer.index:
            return []  # already a valid term

        prefix_matches = [w for w in vocab if w.startswith(term)]
        prefix_matches.sort(
            key=lambda w: len(self._indexer.index[w]), reverse=True
        )

        close = difflib.get_close_matches(term, vocab, n=n * 2, cutoff=0.75)

        seen: set[str] = set()
        suggestions: list[str] = []
        for w in prefix_matches + close:
            if w not in seen:
                seen.add(w)
                suggestions.append(w)
            if len(suggestions) >= n:
                break

        return suggestions

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        """Render a list of SearchResults as a human-readable string."""
        if not results:
            return "No pages found."

        lines = [f"Found {len(results)} page(s):\n"]
        for r in results:
            lines.append(f"  [{r.rank}] {r.url}")
            lines.append(f"       Title   : {r.title}")
            lines.append(f"       Hits    : {r.frequency}  |  TF-IDF: {r.tf_idf_score:.4f}")
            lines.append(f"       Positions (first term): {r.positions}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def format_posting(word: str, entries: list[PageEntry]) -> str:
        """Render a posting list as a human-readable string."""
        if not entries:
            return f"'{word}' not found in index."

        lines = [f"Inverted index for '{word}' ({len(entries)} page(s)):\n"]
        for i, e in enumerate(entries, 1):
            lines.append(f"  [{i}] {e.url}")
            lines.append(f"       Frequency : {e.frequency}")
            lines.append(f"       TF-IDF    : {e.tf_idf:.6f}")
            lines.append(
                f"       Positions : {e.positions[:20]}"
                + (" ..." if len(e.positions) > 20 else "")
            )
            lines.append("")
        return "\n".join(lines)
