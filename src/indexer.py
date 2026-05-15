"""
indexer.py — Inverted index builder and TF-IDF ranker.

Builds an inverted index from crawled pages, storing for each word:
  * which pages contain it
  * its raw frequency per page
  * the positions (word offsets) at which it appears
  * its TF-IDF weight per page (for ranked retrieval)

Design decisions
----------------
* Token normalisation: lower-case, strip non-alpha (keeps hyphens inside
  words), remove pure-punctuation tokens.
* TF-IDF formula (sklearn-style smoothed):
    TF(t,d)  = freq(t,d) / total_tokens(d)
    IDF(t)   = log( (1 + N) / (1 + df(t)) ) + 1
    Weight   = TF * IDF

"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PageEntry:
    """Statistics for one (word, page) pair stored in the inverted index."""

    url: str
    frequency: int
    positions: list[int]   # 0-based word offsets within the page
    tf_idf: float = 0.0    # computed after all pages are indexed


@dataclass
class IndexStats:
    """Summary statistics about the index."""

    total_pages: int = 0
    total_tokens: int = 0
    unique_terms: int = 0
    build_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z](?:[a-zA-Z'\-]*[a-zA-Z])?")


def tokenise(text: str) -> list[str]:
    """
    Tokenise text into lower-case alpha tokens.
    """
    return [m.group().lower() for m in _TOKEN_RE.finditer(text)]


def extract_text(html: str) -> str:
    """
    Extract visible text from an HTML page, stripping scripts/styles.
    Returns a single whitespace-normalised string.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """
    Builds and manages an inverted index over a collection of web pages.
    """

    def __init__(self) -> None:
        self.index: dict[str, list[PageEntry]] = {}
        self.page_metadata: dict[str, dict[str, Any]] = {}
        self.stats = IndexStats()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_page(self, url: str, html: str) -> None:
        """
        Parse html and update the inverted index with tokens found.

        Parameters
        ----------
        url  : Canonical URL of the page.
        html : Raw HTML content.
        """
        text = extract_text(html)
        tokens = tokenise(text)

        if not tokens:
            logger.debug("No tokens extracted from %s — skipping", url)
            return

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Build per-page term frequency + position map
        term_positions: dict[str, list[int]] = {}
        for pos, token in enumerate(tokens):
            term_positions.setdefault(token, []).append(pos)

        # Merge into the global inverted index
        for term, positions in term_positions.items():
            entry = PageEntry(
                url=url,
                frequency=len(positions),
                positions=positions,
            )
            self.index.setdefault(term, []).append(entry)

        self.page_metadata[url] = {
            "title": title,
            "token_count": len(tokens),
        }
        logger.debug(
            "Indexed %s: %d tokens, %d unique terms",
            url, len(tokens), len(term_positions)
        )

    def finalise(self) -> None:
        """
        Compute TF-IDF weights for every (term, page) pair and sort
        each posting list by descending TF-IDF score.
        """
        N = len(self.page_metadata)
        if N == 0:
            return

        for term, entries in self.index.items():
            df = len(entries)
            idf = math.log((1 + N) / (1 + df)) + 1.0  # smoothed IDF

            for entry in entries:
                token_count = self.page_metadata[entry.url]["token_count"]
                tf = entry.frequency / token_count if token_count else 0.0
                entry.tf_idf = round(tf * idf, 6)

            entries.sort(key=lambda e: e.tf_idf, reverse=True)

        self.stats.unique_terms = len(self.index)
        self.stats.total_pages = N
        self.stats.total_tokens = sum(
            m["token_count"] for m in self.page_metadata.values()
        )
        logger.info(
            "Index finalised: %d pages, %d unique terms, %d total tokens",
            self.stats.total_pages,
            self.stats.unique_terms,
            self.stats.total_tokens,
        )

    def build_from_pages(self, pages: list[tuple[str, str]]) -> None:
        """Build index from a list of (url, html) tuples. Used in tests."""
        t0 = time.perf_counter()
        for url, html in pages:
            self.add_page(url, html)
        self.finalise()
        self.stats.build_time_seconds = round(time.perf_counter() - t0, 3)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def find(self, query: str) -> list[PageEntry]:
        """
        Find pages containing ALL query terms (AND semantics).
        Results are ranked by the SUM of TF-IDF scores across all terms..
        """
        terms = tokenise(query)
        if not terms:
            return []

        postings: list[dict[str, PageEntry]] = []
        for term in terms:
            entries = self.index.get(term, [])
            postings.append({e.url: e for e in entries})

        if not postings:
            return []

        # AND intersection
        common_urls = set(postings[0].keys())
        for posting in postings[1:]:
            common_urls &= posting.keys()

        if not common_urls:
            return []

        # Aggregate scores across all query terms
        results: list[PageEntry] = []
        for url in common_urls:
            combined_score = sum(p[url].tf_idf for p in postings if url in p)
            first_entry = postings[0][url]
            result_entry = PageEntry(
                url=url,
                frequency=sum(p[url].frequency for p in postings if url in p),
                positions=first_entry.positions,
                tf_idf=round(combined_score, 6),
            )
            results.append(result_entry)

        results.sort(key=lambda e: e.tf_idf, reverse=True)
        return results

    def print_index(self, word: str) -> list[PageEntry]:
        """
        Return the posting list for word (case-insensitive).
        Returns [] if the word is not in the index.
        """
        term = word.strip().lower()
        return self.index.get(term, [])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the index to a JSON file at path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "stats": {
                "total_pages": self.stats.total_pages,
                "total_tokens": self.stats.total_tokens,
                "unique_terms": self.stats.unique_terms,
                "build_time_seconds": self.stats.build_time_seconds,
            },
            "page_metadata": self.page_metadata,
            "index": {
                term: [
                    {
                        "url": e.url,
                        "frequency": e.frequency,
                        "positions": e.positions,
                        "tf_idf": e.tf_idf,
                    }
                    for e in entries
                ]
                for term, entries in self.index.items()
            },
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

        file_size_kb = path.stat().st_size / 1024
        logger.info("Index saved to %s (%.1f KB)", path, file_size_kb)

    def load(self, path: str | Path) -> None:
        """
        Deserialise the index from a JSON file at path.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if "index" not in payload or "page_metadata" not in payload:
            raise ValueError("Index file is malformed or from an incompatible version.")

        self.page_metadata = payload["page_metadata"]

        self.index = {}
        for term, raw_entries in payload["index"].items():
            self.index[term] = [
                PageEntry(
                    url=e["url"],
                    frequency=e["frequency"],
                    positions=e["positions"],
                    tf_idf=e["tf_idf"],
                )
                for e in raw_entries
            ]

        raw_stats = payload.get("stats", {})
        self.stats = IndexStats(
            total_pages=raw_stats.get("total_pages", 0),
            total_tokens=raw_stats.get("total_tokens", 0),
            unique_terms=raw_stats.get("unique_terms", 0),
            build_time_seconds=raw_stats.get("build_time_seconds", 0.0),
        )

        logger.info(
            "Index loaded from %s: %d terms across %d pages",
            path, self.stats.unique_terms, self.stats.total_pages,
        )
