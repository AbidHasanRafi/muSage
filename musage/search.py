"""Web search module — DuckDuckGo primary, Google fallback (if available)"""

import logging
from typing import List, Dict, Optional

# ── DuckDuckGo (primary — no API key, reliable) ───────────────────────────────
try:
    from ddgs import DDGS as _DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS  # legacy name
    except ImportError:
        _DDGS = None  # type: ignore

# ── Google (optional secondary — may be rate-limited) ────────────────────────
try:
    from googlesearch import search as _google_search
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)

# High-quality reference domains to prefer when ranking results
_PREFERRED_DOMAINS = (
    'wikipedia.org', 'britannica.com', 'britannica.com',
    'stackoverflow.com', 'docs.python.org', 'developer.mozilla.org',
    'github.com', 'reuters.com', 'apnews.com', 'nature.com',
    'sciencedirect.com', 'investopedia.com', 'healthline.com',
    'mayoclinic.org', 'history.com', 'nationalgeographic.com',
)

# Domains that tend to produce noisy scraped content
_NOISY_DOMAINS = (
    'bbc.com', 'bbc.co.uk',  # contains "Image source, Getty Images" captions
)


def _rank(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Rank search results, preferring high-quality reference domains.
    Returns sorted list with preferred domains first.
    """
    def score(item: Dict[str, str]) -> int:
        url = item.get('url', '').lower()
        # Preferred domains get higher score
        for domain in _PREFERRED_DOMAINS:
            if domain in url:
                return 10
        # Noisy domains get lower score
        for domain in _NOISY_DOMAINS:
            if domain in url:
                return -5
        return 0
    
    return sorted(items, key=score, reverse=True)


class WebSearcher:
    """
    Web searcher using DuckDuckGo as the primary backend.
    Falls back to googlesearch-python if DDG is unavailable.
    Results are re-ranked to prefer high-quality reference domains.
    """

    def __init__(self):
        self._ddgs = _DDGS() if _DDGS is not None else None

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """
        Search the web and return ranked results.
        Returns list of dicts: {'title', 'url', 'snippet'}.
        """
        if max_results is None:
            max_results = config.MAX_SEARCH_RESULTS

        results = []

        # 1 — Try DuckDuckGo first (fast, no rate limit issues)
        if self._ddgs is not None:
            results = self._search_ddg(query, max_results)

        # 2 — Fall back to Google if DDG returned nothing
        if not results and _GOOGLE_AVAILABLE:
            results = self._search_google(query, max_results)

        logger.info(f"Search: {len(results)} results for {query!r}")
        return results

    def is_online(self) -> bool:
        """Check if internet is available."""
        if self._ddgs is not None:
            try:
                list(self._ddgs.text('python', max_results=1))
                return True
            except Exception:
                pass
        if _GOOGLE_AVAILABLE:
            try:
                list(_google_search('python', num_results=1))
                return True
            except Exception:
                pass
        return False

    # ── Backends ──────────────────────────────────────────────────────────────

    def _search_ddg(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo text search (primary)."""
        try:
            raw = list(self._ddgs.text(
                query,
                max_results=max_results + 4,
                safesearch='moderate',
                region='wt-wt',
            ))
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

        items = [
            {'title': r.get('title', ''), 'url': r.get('href', ''),
             'snippet': r.get('body', '')}
            for r in raw
        ]
        return _rank(items)[:max_results]

    def _search_google(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Google search via googlesearch-python (fallback)."""
        try:
            raw = list(_google_search(
                query,
                num_results=max_results + 4,
                advanced=True,
                lang='en',
                safe='active',
                sleep_interval=1,
            ))
        except Exception as e:
            logger.warning(f"Google search failed: {e}")
            return []

        items = [
            {'title':   getattr(r, 'title', '') or '',
             'url':     getattr(r, 'url',   '') or '',
             'snippet': getattr(r, 'description', '') or ''}
            for r in raw
        ]
        return _rank(items)[:max_results]
