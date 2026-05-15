"""
main.py — Command-line interface for the COMP3011 Search Engine.

Commands
--------
  build              Crawl the website and build the inverted index.
  load               Load a previously built index from disk.
  print <word>       Print the inverted index entry for a word.
  find <query>       Find pages containing all terms in the query.
  suggest <term>     Show query suggestions for a partial/misspelled term.
  stats              Display index statistics.
  benchmark <query>  Time a find query over 100 runs.
  help               Show this help message.
  exit / quit        Exit the shell.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

from crawler import Crawler, CrawlerConfig
from indexer import Indexer
from search import SearchEngine

DEFAULT_INDEX = Path(__file__).parent.parent / "data" / "index.json"
TARGET_URL = "https://quotes.toscrape.com/"


class SearchShell:
    """Interactive REPL for the search engine."""

    BANNER = (
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║      COMP3011 Search Engine  —  Leeds University     ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
        "Type 'help' for commands, 'exit' to quit.\n"
    )

    HELP = (
        "\nAvailable commands:\n"
        "  build              Crawl website and build the index\n"
        "  load               Load index from disk\n"
        "  print <word>       Print posting list for <word>\n"
        "  find <query>       Find pages matching all query terms\n"
        "  suggest <term>     Query suggestions for <term>\n"
        "  stats              Show index statistics\n"
        "  benchmark <query>  Time a find query\n"
        "  help               Show this help\n"
        "  exit / quit        Exit\n"
    )

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._indexer: Indexer | None = None
        self._engine: SearchEngine | None = None

    def run(self) -> None:
        print(self.BANNER)
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            try:
                self._dispatch(cmd, args)
            except Exception as exc:
                print(f"Error: {exc}")
                logger.exception("Unhandled error in command %r", cmd)

    def _dispatch(self, cmd: str, args: str) -> None:
        match cmd:
            case "build":
                self._cmd_build()
            case "load":
                self._cmd_load()
            case "print":
                self._cmd_print(args)
            case "find":
                self._cmd_find(args)
            case "suggest":
                self._cmd_suggest(args)
            case "stats":
                self._cmd_stats()
            case "benchmark":
                self._cmd_benchmark(args)
            case "help" | "?":
                print(self.HELP)
            case "exit" | "quit" | "q":
                print("Goodbye.")
                sys.exit(0)
            case _:
                print(f"Unknown command '{cmd}'. Type 'help' for options.")

    def _cmd_build(self) -> None:
        print(f"\nBuilding index from {TARGET_URL}")
        print("(This will take several minutes due to the 6-second politeness window)\n")

        config = CrawlerConfig(start_url=TARGET_URL, politeness_window=6.0)
        crawler = Crawler(config)
        indexer = Indexer()

        t0 = time.perf_counter()
        page_count = 0

        for page in crawler.crawl():
            indexer.add_page(page.url, page.html)
            page_count += 1
            print(f"  [{page_count:>3}] {page.url}")

        print(f"\nCrawl complete. {page_count} pages fetched.")
        print("Finalising index (computing TF-IDF weights)...")
        indexer.finalise()
        indexer.stats.build_time_seconds = round(time.perf_counter() - t0, 2)

        print(f"Saving index to {self.index_path}...")
        indexer.save(self.index_path)

        self._indexer = indexer
        self._engine = SearchEngine(indexer)

        print(
            f"\n✓ Index built and saved.\n"
            f"  Pages  : {indexer.stats.total_pages}\n"
            f"  Terms  : {indexer.stats.unique_terms}\n"
            f"  Tokens : {indexer.stats.total_tokens}\n"
            f"  Time   : {indexer.stats.build_time_seconds}s\n"
        )

    def _cmd_load(self) -> None:
        if not self.index_path.exists():
            print(
                f"Index file not found at '{self.index_path}'.\n"
                "Run 'build' first to create the index."
            )
            return

        print(f"Loading index from {self.index_path}...")
        indexer = Indexer()
        indexer.load(self.index_path)
        self._indexer = indexer
        self._engine = SearchEngine(indexer)
        print(
            f"✓ Index loaded.\n"
            f"  Pages  : {indexer.stats.total_pages}\n"
            f"  Terms  : {indexer.stats.unique_terms}\n"
            f"  Tokens : {indexer.stats.total_tokens}\n"
        )

    def _cmd_print(self, args: str) -> None:
        if not self._require_index():
            return
        if not args:
            print("Usage: print <word>")
            return
        word = args.strip().split()[0]
        entries = self._engine.print_word(word)
        print(SearchEngine.format_posting(word, entries))

    def _cmd_find(self, args: str) -> None:
        if not self._require_index():
            return
        if not args:
            print("Usage: find <query>")
            return

        results = self._engine.find(args)
        print(SearchEngine.format_results(results))

        if not results:
            first_term = args.strip().split()[0]
            suggestions = self._engine.suggest(first_term)
            if suggestions:
                print(f"Did you mean: {', '.join(suggestions)} ?")

    def _cmd_suggest(self, args: str) -> None:
        if not self._require_index():
            return
        if not args:
            print("Usage: suggest <term>")
            return
        suggestions = self._engine.suggest(args.strip())
        if suggestions:
            print(f"Suggestions for '{args.strip()}': {', '.join(suggestions)}")
        else:
            print(f"No suggestions for '{args.strip()}'.")

    def _cmd_stats(self) -> None:
        if not self._require_index():
            return
        stats = self._indexer.stats
        print(
            f"\nIndex Statistics\n"
            f"  Total pages   : {stats.total_pages}\n"
            f"  Total tokens  : {stats.total_tokens}\n"
            f"  Unique terms  : {stats.unique_terms}\n"
            f"  Build time    : {stats.build_time_seconds}s\n"
            f"  Avg tokens/pg : {stats.total_tokens // max(1, stats.total_pages)}\n"
        )

    def _cmd_benchmark(self, args: str) -> None:
        if not self._require_index():
            return
        if not args:
            print("Usage: benchmark <query>")
            return

        RUNS = 100
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            self._engine.find(args)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        print(
            f"\nBenchmark: find({args!r}) over {RUNS} runs\n"
            f"  Min : {min(times):.3f} ms\n"
            f"  Avg : {avg:.3f} ms\n"
            f"  Max : {max(times):.3f} ms\n"
        )

    def _require_index(self) -> bool:
        if self._indexer is None or self._engine is None:
            print("Index not loaded. Run 'build' or 'load' first.")
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="search_engine",
        description="COMP3011 Search Engine — command-line interface",
    )
    parser.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX,
        help=f"Path to index file (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    shell = SearchShell(index_path=args.index)
    shell.run()


if __name__ == "__main__":
    main()
