"""
crawler.py — Web crawler for the search engine.

Crawls all pages of a target website, respects a politeness window
between requests, handles network errors gracefully, and returns
raw page content for indexing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    """Represents a single crawled web page."""

    url: str
    html: str
    status_code: int
    crawl_time: float  # seconds taken to fetch


@dataclass
class CrawlerConfig:
    """Configuration for the Crawler."""

    start_url: str = "https://quotes.toscrape.com/"
    politeness_window: float = 6.0  # seconds between requests
    timeout: int = 30               # HTTP request timeout in seconds
    max_retries: int = 3            # retries on transient failures
    user_agent: str = (
        "COMP3011-SearchBot/1.0 (Educational crawler; "
        "University of Leeds; respectful crawl)"
    )
    max_pages: int = 500            # safety cap


class Crawler:
    """
    Breadth-first web crawler that stays within the same domain.

    Design decisions
    ----------------
    * BFS queue ensures pages are crawled in discovery order, giving a
      natural depth-level ordering that aids debugging.
    * The visited set stores normalised URLs (scheme + netloc + path,
      no query string or fragment) to avoid re-crawling the same resource
      via slightly different URLs.
    * A fixed politeness_window throttles requests; the actual sleep is
      max(0, window - elapsed) so network latency already counts.

    Complexity
    ----------
    * Time:  O(P * L) where P = pages crawled, L = links per page.
    * Space: O(P) for visited set + queue (bounded by max_pages).
    """

    def __init__(self, config: CrawlerConfig | None = None) -> None:
        self.config = config or CrawlerConfig()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.config.user_agent})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> Generator[CrawledPage, None, None]:
        """
        Crawl all reachable pages starting from config.start_url.

        Yields
        ------
        CrawledPage
            One object per successfully fetched page.
        """
        base = self._base_url(self.config.start_url)
        queue: list[str] = [self.config.start_url]
        visited: set[str] = set()
        last_request_time: float = 0.0
        pages_crawled = 0

        logger.info("Crawl starting from %s", self.config.start_url)

        while queue and pages_crawled < self.config.max_pages:
            url = queue.pop(0)
            normalised = self._normalise(url)

            if normalised in visited:
                continue
            visited.add(normalised)

            # ── Politeness window ──────────────────────────────────────
            elapsed = time.perf_counter() - last_request_time
            sleep_for = max(0.0, self.config.politeness_window - elapsed)
            if sleep_for > 0:
                logger.debug("Politeness sleep %.2fs before %s", sleep_for, url)
                time.sleep(sleep_for)

            page = self._fetch(url)
            last_request_time = time.perf_counter()

            if page is None:
                continue

            pages_crawled += 1
            logger.info("[%d] Crawled %s (%d)", pages_crawled, url, page.status_code)
            yield page

            # ── Discover links ─────────────────────────────────────────
            for link in self._extract_links(page.html, url, base):
                if self._normalise(link) not in visited:
                    queue.append(link)

        logger.info("Crawl complete. Pages fetched: %d", pages_crawled)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> CrawledPage | None:
        """Fetch a single URL with retry logic. Returns None on failure."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = self._session.get(url, timeout=self.config.timeout)
                elapsed = time.perf_counter() - t0
                resp.raise_for_status()
                return CrawledPage(
                    url=url,
                    html=resp.text,
                    status_code=resp.status_code,
                    crawl_time=elapsed,
                )
            except requests.exceptions.HTTPError as exc:
                logger.warning("HTTP error %s for %s (attempt %d)", exc, url, attempt)
                if exc.response is not None and exc.response.status_code < 500:
                    return None  # 4xx — no point retrying
            except requests.exceptions.ConnectionError as exc:
                logger.warning("Connection error %s for %s (attempt %d)", exc, url, attempt)
            except requests.exceptions.Timeout:
                logger.warning("Timeout for %s (attempt %d)", url, attempt)
            except requests.exceptions.RequestException as exc:
                logger.warning("Request error %s for %s (attempt %d)", exc, url, attempt)

            if attempt < self.config.max_retries:
                time.sleep(2 ** attempt)  # exponential back-off

        logger.error("Giving up on %s after %d attempts", url, self.config.max_retries)
        return None

    def _extract_links(self, html: str, current_url: str, base: str) -> list[str]:
        """
        Extract all same-domain absolute href links from html.

        Parameters
        ----------
        html        : Raw HTML string.
        current_url : URL of the page being parsed (resolves relative links).
        base        : The scheme+netloc we restrict crawling to.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            absolute = urljoin(current_url, href)
            absolute = absolute.split("#")[0]  # strip fragment
            if self._base_url(absolute) == base:
                links.append(absolute)
        return links

    @staticmethod
    def _base_url(url: str) -> str:
        """Return scheme + netloc (e.g. 'https://quotes.toscrape.com')."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _normalise(url: str) -> str:
        """
        Normalise URL for deduplication: lower-case scheme/host,
        strip trailing slash, strip query and fragment.
        """
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
