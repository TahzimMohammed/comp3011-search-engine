"""
test_crawler.py — Unit and integration tests for the Crawler module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import Crawler, CrawlerConfig, CrawledPage


def _make_html(links: list[str] = (), body: str = "") -> str:
    link_tags = "".join(f'<a href="{l}">link</a>' for l in links)
    return f"<html><head><title>Test</title></head><body>{link_tags}{body}</body></html>"


def _mock_response(url: str, html: str = "<html></html>", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.status_code = status
    resp.url = url
    resp.raise_for_status = MagicMock()
    if status >= 400:
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    return resp


class TestNormalise:
    def test_strips_trailing_slash(self):
        assert Crawler._normalise("https://example.com/page/") == "https://example.com/page"

    def test_root_stays_slash(self):
        assert Crawler._normalise("https://example.com/") == "https://example.com/"

    def test_strips_fragment(self):
        assert Crawler._normalise("https://example.com/page#section") == "https://example.com/page"

    def test_lower_cases_scheme_and_host(self):
        assert Crawler._normalise("HTTPS://Example.COM/Page") == "https://example.com/Page"

    def test_preserves_path_case(self):
        result = Crawler._normalise("https://example.com/Page")
        assert "Page" in result


class TestBaseUrl:
    def test_basic(self):
        assert Crawler._base_url("https://quotes.toscrape.com/page/2/") == "https://quotes.toscrape.com"

    def test_subdomain(self):
        assert Crawler._base_url("https://sub.example.com/path") == "https://sub.example.com"


class TestExtractLinks:
    BASE = "https://quotes.toscrape.com"
    CURRENT = "https://quotes.toscrape.com/"

    def _crawl(self) -> Crawler:
        return Crawler(CrawlerConfig(start_url=self.CURRENT))

    def test_relative_link_resolved(self):
        html = _make_html(links=["/page/2/"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert "https://quotes.toscrape.com/page/2/" in links

    def test_absolute_same_domain_kept(self):
        html = _make_html(links=["https://quotes.toscrape.com/author/"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert "https://quotes.toscrape.com/author/" in links

    def test_external_link_excluded(self):
        html = _make_html(links=["https://google.com/"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert not links

    def test_fragment_only_excluded(self):
        html = _make_html(links=["#section"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert not links

    def test_javascript_excluded(self):
        html = _make_html(links=["javascript:void(0)"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert not links

    def test_fragment_stripped_from_link(self):
        html = _make_html(links=["/page/2/#top"])
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert all("#" not in l for l in links)

    def test_empty_href_excluded(self):
        html = '<html><body><a href="">empty</a></body></html>'
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert not links

    def test_no_links(self):
        html = "<html><body><p>No links here</p></body></html>"
        links = self._crawl()._extract_links(html, self.CURRENT, self.BASE)
        assert links == []


class TestFetch:
    def _crawler(self) -> Crawler:
        return Crawler(CrawlerConfig(
            start_url="https://quotes.toscrape.com/",
            politeness_window=0,
            max_retries=3,
        ))

    def test_successful_fetch(self):
        crawler = self._crawler()
        with patch.object(crawler._session, "get") as mock_get:
            mock_get.return_value = _mock_response("https://example.com/", "<html/>")
            page = crawler._fetch("https://example.com/")
        assert page is not None
        assert page.status_code == 200
        assert page.html == "<html/>"

    def test_4xx_returns_none_no_retry(self):
        crawler = self._crawler()
        with patch.object(crawler._session, "get") as mock_get:
            mock_get.return_value = _mock_response("https://example.com/", status=404)
            page = crawler._fetch("https://example.com/")
        assert page is None
        assert mock_get.call_count == 1

    def test_5xx_retries_then_returns_none(self):
        crawler = self._crawler()
        resp = _mock_response("https://example.com/", status=503)
        with patch.object(crawler._session, "get", return_value=resp) as mock_get:
            with patch("time.sleep"):
                page = crawler._fetch("https://example.com/")
        assert page is None
        assert mock_get.call_count == crawler.config.max_retries

    def test_timeout_retries(self):
        crawler = self._crawler()
        with patch.object(crawler._session, "get", side_effect=requests.exceptions.Timeout) as mock_get:
            with patch("time.sleep"):
                page = crawler._fetch("https://example.com/")
        assert page is None
        assert mock_get.call_count == crawler.config.max_retries

    def test_connection_error_retries(self):
        crawler = self._crawler()
        with patch.object(crawler._session, "get", side_effect=requests.exceptions.ConnectionError):
            with patch("time.sleep"):
                page = crawler._fetch("https://example.com/")
        assert page is None

    def test_crawl_time_recorded(self):
        crawler = self._crawler()
        with patch.object(crawler._session, "get") as mock_get:
            mock_get.return_value = _mock_response("https://example.com/")
            page = crawler._fetch("https://example.com/")
        assert page.crawl_time >= 0


class TestPolitenessWindow:
    def test_sleep_called_between_requests(self):
        cfg = CrawlerConfig(
            start_url="https://quotes.toscrape.com/",
            politeness_window=6.0,
            max_pages=2,
        )
        crawler = Crawler(cfg)

        page1_html = _make_html(links=["/page/2/"], body="page one")
        page2_html = _make_html(body="page two")

        responses = [
            _mock_response("https://quotes.toscrape.com/", page1_html),
            _mock_response("https://quotes.toscrape.com/page/2/", page2_html),
        ]

        sleep_calls = []

        def capture_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(crawler._session, "get", side_effect=responses):
            with patch("time.sleep", side_effect=capture_sleep):
                pages = list(crawler.crawl())

        assert any(s > 0 for s in sleep_calls), \
            "Expected at least one non-zero sleep for politeness window"


class TestDeduplication:
    def test_same_url_not_crawled_twice(self):
        cfg = CrawlerConfig(
            start_url="https://quotes.toscrape.com/",
            politeness_window=0,
            max_pages=10,
        )
        crawler = Crawler(cfg)
        html = _make_html(links=["/", "https://quotes.toscrape.com/"])

        with patch.object(crawler._session, "get") as mock_get:
            mock_get.return_value = _mock_response("https://quotes.toscrape.com/", html)
            with patch("time.sleep"):
                pages = list(crawler.crawl())

        assert len(pages) == 1


class TestMaxPages:
    def test_max_pages_respected(self):
        cfg = CrawlerConfig(
            start_url="https://quotes.toscrape.com/",
            politeness_window=0,
            max_pages=3,
        )
        crawler = Crawler(cfg)

        def _make_resp(n: int):
            links = [f"/page/{i}/" for i in range(n, n + 5)]
            return _mock_response(
                f"https://quotes.toscrape.com/page/{n}/",
                _make_html(links=links),
            )

        responses = [_make_resp(i) for i in range(20)]

        with patch.object(crawler._session, "get", side_effect=responses):
            with patch("time.sleep"):
                pages = list(crawler.crawl())

        assert len(pages) <= cfg.max_pages


class TestCrawledPage:
    def test_page_fields_populated(self):
        cfg = CrawlerConfig(
            start_url="https://quotes.toscrape.com/",
            politeness_window=0,
            max_pages=1,
        )
        crawler = Crawler(cfg)
        html = _make_html(body="Hello world")

        with patch.object(crawler._session, "get") as mock_get:
            mock_get.return_value = _mock_response("https://quotes.toscrape.com/", html)
            with patch("time.sleep"):
                pages = list(crawler.crawl())

        assert len(pages) == 1
        page = pages[0]
        assert page.url == "https://quotes.toscrape.com/"
        assert page.status_code == 200
        assert "Hello world" in page.html
        assert isinstance(page.crawl_time, float)
