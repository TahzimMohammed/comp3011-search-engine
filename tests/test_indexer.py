"""
test_indexer.py — Tests for the Indexer and tokeniser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import Indexer, PageEntry, tokenise, extract_text

SIMPLE_HTML = "<html><body><p>The quick brown fox jumps over the lazy dog</p></body></html>"
REPEAT_HTML = "<html><body><p>good good good bad</p></body></html>"


class TestTokenise:
    def test_basic_words(self):
        assert tokenise("hello world") == ["hello", "world"]

    def test_lower_cases(self):
        assert tokenise("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenise("Hello, world!") == ["hello", "world"]

    def test_hyphenated_word_kept(self):
        assert "well-known" in tokenise("well-known fact")

    def test_apostrophe_kept(self):
        assert "it's" in tokenise("it's a test")

    def test_empty_string(self):
        assert tokenise("") == []

    def test_whitespace_only(self):
        assert tokenise("   \n\t  ") == []

    def test_numbers_not_matched(self):
        result = tokenise("42 bottles")
        assert "42" not in result
        assert "bottles" in result


class TestExtractText:
    def test_strips_script_tag(self):
        html = "<html><body><script>alert('x')</script><p>Hello</p></body></html>"
        assert "alert" not in extract_text(html)
        assert "Hello" in extract_text(html)

    def test_strips_style_tag(self):
        html = "<html><head><style>body{color:red}</style></head><body>World</body></html>"
        assert "color" not in extract_text(html)
        assert "World" in extract_text(html)

    def test_whitespace_normalised(self):
        html = "<html><body>  Hello    World  </body></html>"
        assert "  " not in extract_text(html)

    def test_empty_html(self):
        assert extract_text("") == ""


class TestAddPage:
    def test_terms_present_after_add(self):
        idx = Indexer()
        idx.add_page("https://example.com/", SIMPLE_HTML)
        assert "fox" in idx.index
        assert "lazy" in idx.index

    def test_frequency_counted(self):
        idx = Indexer()
        idx.add_page("https://example.com/", REPEAT_HTML)
        assert any(e.frequency == 3 for e in idx.index.get("good", []))

    def test_positions_recorded(self):
        idx = Indexer()
        idx.add_page("https://example.com/", "<html><body>a b a</body></html>")
        entries = idx.index.get("a", [])
        assert entries and entries[0].positions == [0, 2]

    def test_case_insensitive_storage(self):
        idx = Indexer()
        idx.add_page("https://example.com/", "<html><body>Good GOOD good</body></html>")
        assert "good" in idx.index
        assert "Good" not in idx.index

    def test_metadata_stored(self):
        idx = Indexer()
        html = "<html><head><title>My Page</title></head><body>text</body></html>"
        idx.add_page("https://example.com/", html)
        assert idx.page_metadata["https://example.com/"]["title"] == "My Page"

    def test_empty_page_not_added(self):
        idx = Indexer()
        idx.add_page("https://example.com/", "<html><body></body></html>")
        assert "https://example.com/" not in idx.page_metadata

    def test_multi_page_indexing(self):
        idx = Indexer()
        idx.add_page("https://example.com/p1", "<html><body>apple banana</body></html>")
        idx.add_page("https://example.com/p2", "<html><body>apple cherry</body></html>")
        assert len(idx.index["apple"]) == 2


class TestFinalise:
    def test_tfidf_computed(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>apple apple banana</body></html>")
        idx.add_page("https://b.com/", "<html><body>apple cherry</body></html>")
        idx.finalise()
        assert all(e.tf_idf > 0 for e in idx.index["apple"])
        apple_a = next(e.tf_idf for e in idx.index["apple"] if e.url == "https://a.com/")
        apple_b = next(e.tf_idf for e in idx.index["apple"] if e.url == "https://b.com/")
        assert apple_a > apple_b

    def test_posting_list_sorted_desc(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>apple apple apple</body></html>")
        idx.add_page("https://b.com/", "<html><body>apple banana cherry date elderberry</body></html>")
        idx.finalise()
        scores = [e.tf_idf for e in idx.index["apple"]]
        assert scores == sorted(scores, reverse=True)

    def test_stats_populated(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>hello world</body></html>")
        idx.finalise()
        assert idx.stats.total_pages == 1
        assert idx.stats.unique_terms > 0
        assert idx.stats.total_tokens > 0


class TestFind:
    def _built_index(self) -> Indexer:
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>the quick brown fox</body></html>")
        idx.add_page("https://b.com/", "<html><body>fox and hound good friends</body></html>")
        idx.add_page("https://c.com/", "<html><body>good morning everyone</body></html>")
        idx.finalise()
        return idx

    def test_single_term_found(self):
        idx = self._built_index()
        urls = [e.url for e in idx.find("fox")]
        assert "https://a.com/" in urls and "https://b.com/" in urls

    def test_multi_term_and(self):
        idx = self._built_index()
        results = idx.find("fox good")
        assert len(results) == 1 and results[0].url == "https://b.com/"

    def test_unknown_term_returns_empty(self):
        assert self._built_index().find("zzzzz") == []

    def test_empty_query_returns_empty(self):
        assert self._built_index().find("") == []
        assert self._built_index().find("   ") == []

    def test_case_insensitive_find(self):
        idx = self._built_index()
        assert idx.find("FOX") == idx.find("fox")

    def test_results_sorted_by_score(self):
        idx = self._built_index()
        scores = [e.tf_idf for e in idx.find("fox")]
        assert scores == sorted(scores, reverse=True)

    def test_and_semantics_no_overlap(self):
        idx = self._built_index()
        assert idx.find("quick friends") == []


class TestPrintIndex:
    def test_known_word(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>hello world</body></html>")
        idx.finalise()
        entries = idx.print_index("hello")
        assert len(entries) == 1 and entries[0].url == "https://a.com/"

    def test_unknown_word(self):
        idx = Indexer()
        idx.finalise()
        assert idx.print_index("nope") == []

    def test_case_insensitive(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>Hello</body></html>")
        idx.finalise()
        assert idx.print_index("HELLO") == idx.print_index("hello")


class TestPersistence:
    def _sample_index(self) -> Indexer:
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>apple banana cherry</body></html>")
        idx.add_page("https://b.com/", "<html><body>apple date elderberry</body></html>")
        idx.finalise()
        return idx

    def test_save_creates_file(self, tmp_path):
        out = tmp_path / "index.json"
        self._sample_index().save(out)
        assert out.exists() and out.stat().st_size > 0

    def test_round_trip(self, tmp_path):
        original = self._sample_index()
        path = tmp_path / "index.json"
        original.save(path)
        loaded = Indexer()
        loaded.load(path)
        assert set(loaded.index.keys()) == set(original.index.keys())
        assert loaded.stats.total_pages == original.stats.total_pages

    def test_load_preserves_scores(self, tmp_path):
        original = self._sample_index()
        path = tmp_path / "index.json"
        original.save(path)
        loaded = Indexer()
        loaded.load(path)
        for o, l in zip(original.index["apple"], loaded.index["apple"]):
            assert abs(o.tf_idf - l.tf_idf) < 1e-9

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Indexer().load("/nonexistent/path/index.json")

    def test_load_malformed_file_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"wrong_key": 42}')
        with pytest.raises(ValueError):
            Indexer().load(bad)

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "index.json"
        self._sample_index().save(nested)
        assert nested.exists()


class TestEdgeCases:
    def test_very_long_page(self):
        html = f"<html><body>{' '.join(['word'] * 10000)}</body></html>"
        idx = Indexer()
        idx.add_page("https://a.com/", html)
        assert idx.index["word"][0].frequency == 10000

    def test_duplicate_url_no_crash(self):
        idx = Indexer()
        idx.add_page("https://a.com/", "<html><body>alpha beta</body></html>")
        idx.add_page("https://a.com/", "<html><body>alpha beta</body></html>")
        idx.finalise()
        assert "alpha" in idx.index
