"""Tests for Web Search with domain safety filtering."""

from unittest.mock import patch, MagicMock

import pytest

from src.rag.web_search import (
    DomainFilter,
    WebSearcher,
    SearchResult,
    WebSearchResponse,
    DEFAULT_ALLOWED_TLDS,
    DEFAULT_ALLOWED_DOMAINS,
)


# ──────────────────────────────────────────────
# DomainFilter — Allowlist Logic
# ──────────────────────────────────────────────

class TestDomainFilter:
    def setup_method(self):
        self.f = DomainFilter()

    # --- Allowed TLDs ---

    def test_allows_gov(self):
        assert self.f.is_allowed("https://www.nasa.gov/news") is True

    def test_allows_edu(self):
        assert self.f.is_allowed("https://cs.stanford.edu/courses") is True

    def test_allows_org(self):
        assert self.f.is_allowed("https://en.wikipedia.org/wiki/Python") is True

    def test_allows_net(self):
        assert self.f.is_allowed("https://docs.somesite.net/guide") is True

    # --- Allowed specific domains ---

    def test_allows_reddit(self):
        assert self.f.is_allowed("https://www.reddit.com/r/python") is True

    def test_allows_reddit_subdomain(self):
        assert self.f.is_allowed("https://old.reddit.com/r/learnpython") is True

    def test_allows_stackoverflow(self):
        assert self.f.is_allowed("https://stackoverflow.com/questions/123") is True

    def test_allows_github(self):
        assert self.f.is_allowed("https://github.com/torvalds/linux") is True

    def test_allows_github_subdomain(self):
        assert self.f.is_allowed("https://docs.github.com/en/actions") is True

    # --- Blocked domains ---

    def test_blocks_random_com(self):
        assert self.f.is_allowed("https://malware-site.com/bad") is False

    def test_blocks_random_io(self):
        assert self.f.is_allowed("https://sketchy.io/phishing") is False

    def test_blocks_random_xyz(self):
        assert self.f.is_allowed("https://spam.xyz/click") is False

    def test_blocks_co_uk(self):
        assert self.f.is_allowed("https://news.bbc.co.uk/article") is False

    # --- Edge cases ---

    def test_empty_url(self):
        assert self.f.is_allowed("") is False

    def test_invalid_url(self):
        assert self.f.is_allowed("not a url at all") is False

    def test_no_scheme(self):
        # urlparse without scheme puts it all in path
        assert self.f.is_allowed("reddit.com/r/python") is False

    def test_http_still_allowed(self):
        assert self.f.is_allowed("http://en.wikipedia.org/wiki/Test") is True

    # --- Custom allowlists ---

    def test_custom_tld(self):
        f = DomainFilter(allowed_tlds=[".uk"], allowed_domains=[])
        assert f.is_allowed("https://bbc.co.uk/news") is True
        assert f.is_allowed("https://reddit.com") is False

    def test_custom_domain(self):
        f = DomainFilter(allowed_tlds=[], allowed_domains=["example.com"])
        assert f.is_allowed("https://example.com/page") is True
        assert f.is_allowed("https://other.com/page") is False

    def test_tld_without_dot_prefix(self):
        f = DomainFilter(allowed_tlds=["gov"], allowed_domains=[])
        assert f.is_allowed("https://nasa.gov") is True

    # --- filter_results ---

    def test_filter_results(self):
        results = [
            SearchResult(title="Good", url="https://python.org/docs", snippet="Python docs"),
            SearchResult(title="Bad", url="https://malware.com/bad", snippet="Bad site"),
            SearchResult(title="Also good", url="https://reddit.com/r/test", snippet="Reddit"),
        ]
        allowed, removed = self.f.filter_results(results)
        assert len(allowed) == 2
        assert removed == 1
        assert all("malware" not in r.url for r in allowed)

    def test_filter_results_all_blocked(self):
        results = [
            SearchResult(title="Bad1", url="https://bad.com", snippet=""),
            SearchResult(title="Bad2", url="https://evil.io", snippet=""),
        ]
        allowed, removed = self.f.filter_results(results)
        assert len(allowed) == 0
        assert removed == 2

    def test_filter_results_all_allowed(self):
        results = [
            SearchResult(title="Gov", url="https://data.gov/dataset", snippet=""),
            SearchResult(title="Edu", url="https://mit.edu/course", snippet=""),
        ]
        allowed, removed = self.f.filter_results(results)
        assert len(allowed) == 2
        assert removed == 0


# ──────────────────────────────────────────────
# WebSearcher — Search + Filter Integration
# ──────────────────────────────────────────────

# Mock DuckDuckGo results simulating a real search
MOCK_DDG_RESULTS = [
    {"title": "Python - Wikipedia", "href": "https://en.wikipedia.org/wiki/Python", "body": "Python is a programming language."},
    {"title": "Python on Reddit", "href": "https://www.reddit.com/r/python", "body": "The Python subreddit."},
    {"title": "Sketchy Site", "href": "https://malware-download.com/python", "body": "Download Python here (not really)."},
    {"title": "Python Docs", "href": "https://docs.python.org", "body": "Official Python documentation."},  # .org
    {"title": "Random Blog", "href": "https://someblog.com/python-tips", "body": "Python tips and tricks."},
    {"title": "NSF Research", "href": "https://nsf.gov/python-in-science", "body": "Python in scientific research."},
    {"title": "Stack Overflow", "href": "https://stackoverflow.com/questions/python", "body": "Python questions."},
    {"title": "GitHub Repo", "href": "https://github.com/python/cpython", "body": "CPython source code."},
]


def _mock_ddgs_text(query, max_results=10):
    """Simulate DuckDuckGo returning results."""
    return MOCK_DDG_RESULTS[:max_results]


class TestWebSearcher:
    @patch("duckduckgo_search.DDGS")
    def test_search_filters_domains(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.return_value = MOCK_DDG_RESULTS
        mock_ddgs_class.return_value = mock_instance

        searcher = WebSearcher(max_results=5)
        response = searcher.search("Python programming")

        # Should have filtered out malware-download.com and someblog.com
        urls = [r.url for r in response.results]
        assert all("malware" not in u for u in urls)
        assert all("someblog" not in u for u in urls)
        assert response.filtered_count == 2

    @patch("duckduckgo_search.DDGS")
    def test_search_keeps_allowed_sites(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.return_value = MOCK_DDG_RESULTS
        mock_ddgs_class.return_value = mock_instance

        searcher = WebSearcher(max_results=10)
        response = searcher.search("Python")

        urls = [r.url for r in response.results]
        assert "https://en.wikipedia.org/wiki/Python" in urls     # .org
        assert "https://www.reddit.com/r/python" in urls          # reddit.com
        assert "https://nsf.gov/python-in-science" in urls        # .gov
        assert "https://stackoverflow.com/questions/python" in urls  # stackoverflow
        assert "https://github.com/python/cpython" in urls        # github

    @patch("duckduckgo_search.DDGS")
    def test_search_respects_max_results(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.return_value = MOCK_DDG_RESULTS
        mock_ddgs_class.return_value = mock_instance

        searcher = WebSearcher(max_results=2)
        response = searcher.search("Python")
        assert len(response.results) <= 2

    @patch("duckduckgo_search.DDGS")
    def test_search_query_preserved(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_ddgs_class.return_value = mock_instance

        searcher = WebSearcher()
        response = searcher.search("test query")
        assert response.query == "test query"

    @patch("duckduckgo_search.DDGS")
    def test_search_handles_ddg_failure(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.side_effect = Exception("Network error")
        mock_ddgs_class.return_value = mock_instance

        searcher = WebSearcher()
        response = searcher.search("test")
        assert response.results == []
        assert response.filtered_count == 0


# ──────────────────────────────────────────────
# LLM Formatting
# ──────────────────────────────────────────────

class TestFormatForLLM:
    def test_format_with_results(self):
        response = WebSearchResponse(
            query="Python",
            results=[
                SearchResult(title="Python.org", url="https://python.org", snippet="Official site."),
                SearchResult(title="Reddit", url="https://reddit.com/r/python", snippet="Python sub."),
            ],
            filtered_count=0,
        )
        searcher = WebSearcher()
        text = searcher.format_for_llm(response)
        assert "Python" in text
        assert "[1]" in text
        assert "[2]" in text
        assert "python.org" in text
        assert "reddit.com" in text

    def test_format_no_results(self):
        response = WebSearchResponse(query="xyzzy", results=[], filtered_count=5)
        searcher = WebSearcher()
        text = searcher.format_for_llm(response)
        assert "No results found" in text
        assert "xyzzy" in text


# ──────────────────────────────────────────────
# Tool Integration
# ──────────────────────────────────────────────

class TestToolIntegration:
    @patch("duckduckgo_search.DDGS")
    def test_tool_handler_returns_formatted_text(self, mock_ddgs_class):
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": "Result", "href": "https://python.org", "body": "A result."},
        ]
        mock_ddgs_class.return_value = mock_instance

        from src.tools.dispatch import _web_search_handler
        output = _web_search_handler("Python")
        assert "python.org" in output

    def test_tool_registered_in_dispatcher(self):
        from src.tools.dispatch import create_default_dispatcher
        dispatcher = create_default_dispatcher()
        assert "web_search" in dispatcher.registry.names

    def test_tool_in_system_prompt(self):
        from src.tools.dispatch import create_default_dispatcher
        dispatcher = create_default_dispatcher()
        prompt = dispatcher.registry.system_prompt_section()
        assert "web_search" in prompt
        assert "trusted domains" in prompt
