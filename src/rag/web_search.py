"""Web Search — Query live search APIs with domain safety filtering.

Uses DuckDuckGo as the default search engine (free, no API key).
Results are filtered through a domain allowlist to prevent the model
from accessing potentially harmful sites.

Default allowed domains:
  - TLDs: .gov, .net, .org, .edu
  - Specific sites: reddit.com, stackoverflow.com, github.com

The allowlist is configurable via model_config.yaml.

Usage:
    searcher = WebSearcher()
    results = searcher.search("Python programming tutorials")
    for r in results:
        print(f"{r.title} — {r.url}")
        print(f"  {r.snippet}")
"""

from dataclasses import dataclass, field
from urllib.parse import urlparse


# ──────────────────────────────────────────────
# Default Safety Allowlist
# ──────────────────────────────────────────────

DEFAULT_ALLOWED_TLDS = [".gov", ".net", ".org", ".edu"]

DEFAULT_ALLOWED_DOMAINS = [
    "reddit.com",
    "stackoverflow.com",
    "github.com",
]


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str


@dataclass
class WebSearchResponse:
    """Response from a web search query."""
    query: str
    results: list[SearchResult]
    filtered_count: int  # How many results were removed by the allowlist


# ──────────────────────────────────────────────
# Domain Filter
# ──────────────────────────────────────────────

class DomainFilter:
    """Filter URLs against an allowlist of TLDs and specific domains.

    Args:
        allowed_tlds: List of allowed top-level domains (e.g. [".gov", ".edu"]).
        allowed_domains: List of specific allowed domains (e.g. ["reddit.com"]).
    """

    def __init__(
        self,
        allowed_tlds: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ):
        self.allowed_tlds = [
            tld if tld.startswith(".") else f".{tld}"
            for tld in (DEFAULT_ALLOWED_TLDS if allowed_tlds is None else allowed_tlds)
        ]
        self.allowed_domains = [
            d.lower()
            for d in (DEFAULT_ALLOWED_DOMAINS if allowed_domains is None else allowed_domains)
        ]

    def is_allowed(self, url: str) -> bool:
        """Check if a URL passes the domain allowlist."""
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
        except Exception:
            return False

        if not hostname:
            return False

        # Check specific allowed domains (including subdomains)
        for domain in self.allowed_domains:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return True

        # Check allowed TLDs
        for tld in self.allowed_tlds:
            if hostname.endswith(tld):
                return True

        return False

    def filter_results(self, results: list[SearchResult]) -> tuple[list[SearchResult], int]:
        """Filter search results, returning (allowed, removed_count)."""
        allowed = [r for r in results if self.is_allowed(r.url)]
        removed = len(results) - len(allowed)
        return allowed, removed


# ──────────────────────────────────────────────
# Web Searcher
# ──────────────────────────────────────────────

class WebSearcher:
    """Search the web with domain safety filtering.

    Args:
        max_results: Maximum results to return (after filtering).
        allowed_tlds: Override default allowed TLDs.
        allowed_domains: Override default allowed domains.
    """

    def __init__(
        self,
        max_results: int = 3,
        allowed_tlds: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ):
        self.max_results = max_results
        self.domain_filter = DomainFilter(
            allowed_tlds=allowed_tlds,
            allowed_domains=allowed_domains,
        )

    def search(self, query: str) -> WebSearchResponse:
        """Search the web and return filtered results.

        Fetches more results than needed from DuckDuckGo, then filters
        through the domain allowlist and returns up to max_results.

        Args:
            query: The search query string.

        Returns:
            WebSearchResponse with filtered results.
        """
        # Fetch extra results since filtering will remove some
        fetch_count = self.max_results * 4

        raw_results = self._fetch_duckduckgo(query, fetch_count)

        # Apply domain filter
        allowed, filtered_count = self.domain_filter.filter_results(raw_results)

        # Limit to max_results
        return WebSearchResponse(
            query=query,
            results=allowed[:self.max_results],
            filtered_count=filtered_count,
        )

    def _fetch_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Fetch results from DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search is required for web search. "
                "Install it: pip install duckduckgo-search"
            )

        try:
            raw = DDGS().text(query, max_results=max_results)
        except Exception:
            return []

        results = []
        for item in raw:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
            ))
        return results

    def format_for_llm(self, response: WebSearchResponse) -> str:
        """Format search results into a text block for the LLM.

        Used by the tool handler to inject results into the conversation.
        """
        if not response.results:
            return f"No results found for: {response.query}"

        parts = [f"Web search results for: {response.query}\n"]
        for i, result in enumerate(response.results, 1):
            parts.append(
                f"[{i}] {result.title}\n"
                f"    URL: {result.url}\n"
                f"    {result.snippet}"
            )
        return "\n\n".join(parts)
