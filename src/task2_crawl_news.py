"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
from html import unescape
from html.parser import HTMLParser
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
REQUEST_TIMEOUT = (10, 30)
MIN_CONTENT_LENGTH = 500
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
    "https://baochinhphu.vn/khoi-to-bat-tam-giam-ca-si-long-nhat-son-ngoc-minh-vi-to-chuc-su-dung-ma-tuy-102260520125739676.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://danviet.vn/nhuc-nhoi-loat-nghe-si-vuong-lao-ly-vi-ma-tuy-khong-chi-la-sa-nga-ma-con-la-su-ton-thuong-doi-voi-niem-tin-cong-chung-d1428424.html",
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-pha-huy-nao-bo-nguoi-tre-ra-sao-2518285.html"
]


class ArticleHTMLParser(HTMLParser):
    """Parser tối giản để lấy nội dung chính khi trang không có JSON-LD."""

    CONTENT_TAGS = {"h1", "h2", "p"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._current_tag: Optional[str] = None
        self._buffer: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth == 0 and tag in self.CONTENT_TAGS:
            self._current_tag = tag
            self._buffer = []

    def handle_data(self, data: str):
        if self._skip_depth == 0 and self._current_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return

        if tag == self._current_tag:
            text = clean_text(" ".join(self._buffer))
            if len(text) >= 25 or tag in {"h1", "h2"}:
                self.paragraphs.append(text)
            self._current_tag = None
            self._buffer = []


def clean_text(text: str) -> str:
    """Chuẩn hoá khoảng trắng và HTML entities."""
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def markdown_from_paragraphs(paragraphs: list[str]) -> str:
    """Ghép các đoạn text thành markdown đơn giản."""
    seen = set()
    unique_paragraphs = []
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if not paragraph or paragraph in seen:
            continue
        seen.add(paragraph)
        unique_paragraphs.append(paragraph)

    return "\n\n".join(unique_paragraphs)


def iter_json_objects(data: Any):
    """Duyệt đệ quy JSON-LD vì nhiều báo bọc article trong @graph."""
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from iter_json_objects(value)
    elif isinstance(data, list):
        for item in data:
            yield from iter_json_objects(item)


def get_jsonld_articles(html: str) -> list[dict]:
    """Trích các object Article/NewsArticle từ script application/ld+json."""
    scripts = re.findall(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    articles = []
    for script in scripts:
        raw_json = unescape(script).strip()
        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        for item in iter_json_objects(data):
            article_type = item.get("@type") or item.get("type") or ""
            if isinstance(article_type, list):
                article_type = " ".join(str(value) for value in article_type)
            if "article" in str(article_type).lower() or item.get("articleBody"):
                articles.append(item)

    return articles


def extract_meta_content(html: str, key: str) -> str:
    """Lấy content của meta name/property theo key."""
    patterns = [
        rf"<meta[^>]+(?:property|name)=[\"']{re.escape(key)}[\"'][^>]+content=[\"']([^\"']+)[\"']",
        rf"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+(?:property|name)=[\"']{re.escape(key)}[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_text(match.group(1))
    return ""


def extract_title(html: str, article: Optional[dict] = None) -> str:
    """Trích title ưu tiên JSON-LD, sau đó tới OpenGraph và thẻ title."""
    article = article or {}
    for value in (article.get("headline"), article.get("name")):
        title = clean_text(str(value or ""))
        if title:
            return title

    for key in ("og:title", "twitter:title"):
        title = extract_meta_content(html, key)
        if title:
            return title

    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return clean_text(match.group(1))

    return "Unknown"


def extract_content_from_html(html: str) -> tuple[str, str, str]:
    """Trích title, ngày publish và nội dung markdown từ HTML."""
    jsonld_articles = get_jsonld_articles(html)
    best_article = max(
        jsonld_articles,
        key=lambda item: len(clean_text(str(item.get("articleBody") or item.get("description") or ""))),
        default={},
    )

    content = clean_text(str(best_article.get("articleBody") or ""))
    if content:
        paragraphs = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỵ0-9\"'])", content)
        content_markdown = markdown_from_paragraphs(paragraphs)
    else:
        parser = ArticleHTMLParser()
        parser.feed(html)
        content_markdown = markdown_from_paragraphs(parser.paragraphs)

    title = extract_title(html, best_article)
    published_at = clean_text(
        str(
            best_article.get("datePublished")
            or best_article.get("dateCreated")
            or extract_meta_content(html, "article:published_time")
            or extract_meta_content(html, "pubdate")
        )
    )

    return title, published_at, content_markdown


def stringify_crawl4ai_markdown(markdown: Any) -> str:
    """Crawl4AI có vài phiên bản trả markdown là str hoặc object."""
    if isinstance(markdown, str):
        return markdown.strip()

    for attr in ("fit_markdown", "raw_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return clean_text(str(markdown))


async def crawl_with_crawl4ai(url: str) -> Optional[dict]:
    """Crawl bằng Crawl4AI nếu package đã được cài."""
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        print(f"  ⚠ Crawl4AI lỗi, chuyển sang requests fallback: {exc}")
        return None

    content_markdown = stringify_crawl4ai_markdown(getattr(result, "markdown", ""))
    metadata = getattr(result, "metadata", None) or {}
    title = clean_text(metadata.get("title", "")) or "Unknown"

    if len(content_markdown) < MIN_CONTENT_LENGTH:
        print("  ⚠ Crawl4AI trả nội dung quá ngắn, chuyển sang requests fallback")
        return None

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "date_published": clean_text(metadata.get("published_time", "")),
        "crawler": "crawl4ai",
        "content": content_markdown,
        "content_markdown": content_markdown,
    }


def crawl_with_requests(url: str) -> dict:
    """Fallback crawler dùng requests và parser nội bộ."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    if response.encoding:
        response.encoding = response.apparent_encoding or response.encoding

    title, published_at, content_markdown = extract_content_from_html(response.text)
    if len(content_markdown) < MIN_CONTENT_LENGTH:
        raise ValueError(
            f"Nội dung crawl quá ngắn ({len(content_markdown)} ký tự), cần kiểm tra lại URL: {url}"
        )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "date_published": published_at,
        "crawler": "requests",
        "content": content_markdown,
        "content_markdown": content_markdown,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    article = await crawl_with_crawl4ai(url)
    if article:
        return article

    return crawl_with_requests(url)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
