"""Advanced web scraping module with robust error handling"""

import re
import time
import logging
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException, Timeout, ConnectionError

from . import config

logger = logging.getLogger(__name__)


class RobustWebScraper:
    """
    A production-grade web scraper with:
    - robots.txt compliance
    - Rate limiting
    - Multiple retry strategies
    - Content extraction intelligence
    - Error recovery
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.last_request_time = {}
        self.robots_cache = {}

    def _check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            if base_url in self.robots_cache:
                return self.robots_cache[base_url]

            robots_url = urljoin(base_url, "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            can_fetch = rp.can_fetch(config.USER_AGENT, url)
            self.robots_cache[base_url] = can_fetch
            return can_fetch
        except Exception as e:
            logger.warning(f"Could not check robots.txt: {e}. Proceeding cautiously.")
            return True  # If we can't check, assume it's okay but be respectful

    def _rate_limit(self, domain: str):
        """Implement rate limiting per domain"""
        current_time = time.time()
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < config.RATE_LIMIT_DELAY:
                time.sleep(config.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time[domain] = time.time()

    def _make_request(self, url: str, retries: int = 0) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        try:
            # Check robots.txt
            if not self._check_robots_txt(url):
                logger.warning(f"Robots.txt disallows scraping: {url}")
                return None

            # Rate limiting
            domain = urlparse(url).netloc
            self._rate_limit(domain)

            # Make request
            response = self.session.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response

        except Timeout:
            if retries < config.MAX_RETRIES:
                logger.info(f"Timeout, retrying... ({retries + 1}/{config.MAX_RETRIES})")
                time.sleep(2 ** retries)  # Exponential backoff
                return self._make_request(url, retries + 1)
            logger.debug(f"Failed after {config.MAX_RETRIES} retries: {url}")
            return None

        except ConnectionError as e:
            logger.debug(f"Connection error for {url}: {e}")
            return None

        except RequestException as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Intelligently extract main content from HTML
        Removes navigation, ads, scripts, etc.
        """
        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        # Try to find main content area
        main_content = None

        # Strategy 1: Look for main/article tags
        for tag_name in ["main", "article", "div[role='main']"]:
            main_content = soup.find(tag_name)
            if main_content:
                break

        # Strategy 2: Look for content-rich divs
        if not main_content:
            content_divs = soup.find_all("div", class_=re.compile(r"(content|main|article|post|entry)", re.I))
            if content_divs:
                main_content = max(content_divs, key=lambda x: len(x.get_text(strip=True)))

        # Strategy 3: Use body if nothing found
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text
        text = main_content.get_text(separator="\n", strip=True)

        # Clean up text
        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text, removing image captions, ads and navigation noise."""
        # ── Image / media captions (BBC, Guardian, etc.) ──────────────────────
        # Remove entire lines that are image source / caption labels
        text = re.sub(r'^\s*Image\s+(source|caption)[,:].*$', '', text,
                      flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*(Getty Images?|AFP|Reuters|AP Photo|PA Media|'  
                      r'Alamy|Shutterstock|EPA)[,\s].*$', '', text,
                      flags=re.MULTILINE | re.IGNORECASE)
        # Inline forms: "Image source, Getty Images" inside a paragraph
        text = re.sub(
            r'Image\s+(source|caption)[,:]\s*[^.\n]{0,100}',
            '', text, flags=re.IGNORECASE)
        text = re.sub(
            r'\b(Getty Images?|AFP|Reuters|AP Photo|PA Media|Alamy|Shutterstock|EPA)\b[,\s]*',
            '', text, flags=re.IGNORECASE)

        # ── Common web noise ──────────────────────────────────────────────────
        text = re.sub(r'Cookie Policy.*?Accept', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Subscribe to.*?Newsletter', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[?Read\s+more[:\s].*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*(Share|Tweet|Email|Print|Save)\s+(this|article|story).*$',
                      '', text, flags=re.MULTILINE | re.IGNORECASE)
        # HTML artefacts that survive parsing
        text = re.sub(r'<[^>]{1,80}>', ' ', text)  # stray tags
        text = re.sub(r'&[a-z]{2,6};', ' ', text)  # HTML entities

        # ── Whitespace normalisation ───────────────────────────────────────────
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # max one blank line
        text = re.sub(r' +', ' ', text)

        # ── Truncate ──────────────────────────────────────────────────────────
        if len(text) > config.MAX_SCRAPE_LENGTH:
            text = text[:config.MAX_SCRAPE_LENGTH] + '...'

        return text.strip()

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from page"""
        metadata = {
            "url": url,
            "title": "",
            "description": "",
        }

        # Extract title
        if soup.title:
            metadata["title"] = soup.title.string.strip() if soup.title.string else ""
        elif soup.find("h1"):
            metadata["title"] = soup.find("h1").get_text(strip=True)

        # Extract description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"].strip()
        elif soup.find("meta", attrs={"property": "og:description"}):
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            metadata["description"] = og_desc.get("content", "").strip()

        return metadata

    def scrape(self, url: str) -> Optional[Dict[str, str]]:
        """
        Main scraping method
        Returns a dict with 'content', 'title', 'description', 'url'
        """
        logger.info(f"Scraping: {url}")

        response = self._make_request(url)
        if not response:
            return None

        try:
            # Parse HTML
            soup = BeautifulSoup(response.content, "lxml")

            # Extract content and metadata
            content = self._extract_main_content(soup)
            metadata = self._extract_metadata(soup, url)

            return {
                "content": content,
                "title": metadata["title"],
                "description": metadata["description"],
                "url": metadata["url"],
            }

        except Exception as e:
            logger.debug(f"Error parsing {url}: {e}")
            return None

    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, str]]:
        """Scrape multiple URLs"""
        results = []
        for url in urls:
            result = self.scrape(url)
            if result:
                results.append(result)
        return results
