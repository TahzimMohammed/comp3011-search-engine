"""
test_search.py — Tests for the SearchEngine and formatting utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import Indexer
from search import SearchEngine, SearchResult


def _build_engine(pages=None) -> SearchEngine:
    if pages is None:
        pages = [
            ("https://a.com/", "<html><head><title>Page A</title></head><body>the quick brown fox jumps</body></html>"),
            ("https://b.com/", "<html><head><title>Page B</title></head><body>fox and hound are good friends</body></html>"),
            ("https://c.com/", "<html><head><title>Page C</title></head><body>good morning world</body></html>"),
            ("https://d.com/", "<html><head><title>Page D</title></head><body>indifference is the enemy of love</body></html>"),
        ]
    idx = Indexer()
    for url, html in pages:
        idx.add_page(url, html)
    idx.finalise()
    return SearchEngine(idx)


class TestFind:
    def test_single_word_returns_results(self):
        results = _build_engine().find("fox")
        assert len(results) >= 1 and all(isinstance(r, SearchResult) for r in results)

    def test_multi_word_and_semantics(self):
        results = _build_engine().find("fox good")
        urls = [r.url for r in results]
        assert "https://b.com/" in urls and "https://a.com/" not in urls

    def test_empty_query_returns_empty(self):
        assert _build_engine().find("") == []
        assert _build_engine().find("   ") == []

    def test_unknown_term_returns_empty(self):
        assert _build_engine().find("zzzzunknownzzz") == []

    def test_case_insensitive(self):
        engine = _build_engine()
        assert [r.url for r in engine.find("FOX")] == [r.url for r in engine.find("fox")]

    def test_results_ranked_by_score(self):
        results = _build_engine().find("fox")
        scores = [r.tf_idf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_title(self):
        assert all(r.title for r in _build_engine().find("fox"))

    def test_result_rank_sequential(self):
        results = _build_engine().find("fox")
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_indifference_search(self):
        results = _build_engine().find("indifference")
        assert len(results) == 1 and results[0].url == "https://d.com/"


class TestPrintWord:
    def test_known_word_returns_entries(self):
        assert len(_build_engine().print_word("fox")) >= 1

    def test_unknown_word_returns_empty(self):
        assert _build_engine().print_word("zzznonsense") == []

    def test_empty_string_returns_empty(self):
        assert _build_engine().print_word("") == []

    def test_whitespace_only_returns_empty(self):
        assert _build_engine().print_word("   ") == []

    def test_case_insensitive(self):
        engine = _build_engine()
        assert engine.print_word("FOX") == engine.print_word("fox")


class TestSuggest:
    def test_exact_match_returns_empty(self):
        assert _build_engine().suggest("fox") == []

    def test_prefix_match(self):
        assert "friends" in _build_engine().suggest("fri")

    def test_empty_query_returns_empty(self):
        assert _build_engine().suggest("") == []

    def test_suggestion_count_capped(self):
        assert len(_build_engine().suggest("a", n=3)) <= 3

    def test_no_crash_on_garbage(self):
        assert isinstance(_build_engine().suggest("zzzzxxxxxqqqqq"), list)


class TestFormatResults:
    def test_empty_list(self):
        assert "No pages found" in SearchEngine.format_results([])

    def test_contains_rank(self):
        results = _build_engine().find("fox")
        assert "[1]" in SearchEngine.format_results(results)

    def test_contains_url(self):
        results = _build_engine().find("fox")
        assert "https://" in SearchEngine.format_results(results)


class TestFormatPosting:
    def test_empty_returns_not_found(self):
        assert "not found" in SearchEngine.format_posting("nonsense", []).lower()

    def test_non_empty_contains_word(self):
        entries = _build_engine().print_word("fox")
        assert "fox" in SearchEngine.format_posting("fox", entries).lower()

    def test_non_empty_contains_frequency(self):
        entries = _build_engine().print_word("fox")
        assert "Frequency" in SearchEngine.format_posting("fox", entries)


class TestBenchmark:
    def test_find_stable_across_runs(self):
        engine = _build_engine()
        assert [r.url for r in engine.find("fox")] == [r.url for r in engine.find("fox")]

    def test_find_speed_reasonable(self):
        import time
        engine = _build_engine()
        t0 = time.perf_counter()
        for _ in range(1000):
            engine.find("fox")
        assert time.perf_counter() - t0 < 2.0
