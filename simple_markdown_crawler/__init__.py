import logging
import queue
import random
import re
import threading
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

__version__ = "0.3.0"
__author__ = "Jose Maldonado (github.com/yukiteruamano)"
__copyright__ = "(C) 2023 Paul Pierre. MIT License. 2025 Jose Maldonado. MIT License"
__contributors__ = ["Jose Maldonado"]

BANNER = r"""
                |                                     |
 __ `__ \    _` |        __|   __|   _` | \ \  \   /  |   _ \   __|
 |   |   |  (   |       (     |     (   |  \ \  \ /   |   __/  |
_|  _|  _| \__._|      \___| _|    \__._|   \_/\_/   _| \___| _|

----------------------------------------------------------------------------
A multithreaded 🕸️ web crawler that recursively crawls a website and
creates a 🔽 markdown file for each page by https://github.com/yukiteruamano
This is a fork from markdown-crawler created by Paul Pierre (abandoned)
----------------------------------------------------------------------------
"""

logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = "markdown"
DEFAULT_MAX_DEPTH = 1
DEFAULT_NUM_THREADS = 1
DEFAULT_TARGET_CONTENT = ["article", "div", "main", "p", "table", "pre"]
DEFAULT_TARGET_LINKS = ["body"]
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True
MAX_NESTING_DEPTH = 400

# Tags whose content is never rendered by browsers and should be dropped
UNRENDERED_TAGS = [
    "script",
    "style",
    "template",
    "noscript",
    "iframe",
    "audio",
    "canvas",
    "svg",
]


# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    if not isinstance(url, str) or url != url.strip():
        return False
    try:
        result = urllib.parse.urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        # Reject whitespace/control characters inside the URL
        if any(char.isspace() or ord(char) < 32 for char in url):
            return False
        # Validate the port, if present (accessing .port raises for out-of-range)
        _ = result.port
    except ValueError:
        logger.debug("❌ Invalid URL %s", url)
        return False
    else:
        return True


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            None,
            parsed.query,
            None,
        )
    )


# ------------------
# HTML parsing logic
# ------------------

# User Agents List
user_agents = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:153.0) "
        "Gecko/20100101 Firefox/153.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:153.0) "
        "Gecko/20100101 Firefox/153.0"
    ),
    (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7_8 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 "
        "Mobile/15E148 Safari/604.1"
    ),
    (
        "Mozilla/5.0 (iPad; CPU OS 18_7_8 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/26.5 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Mobile Safari/537.36"
    ),
    (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/151.0.7922.112 "
        "Mobile/15E148 Safari/604.1"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/151.0.4129.59"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 OPR/136.0.0.0"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Vivaldi/8.1.4087.46"
    ),
]


def crawl(
    url: str,
    base_url: str,
    already_crawled: set,
    already_crawled_lock: threading.Lock,
    file_path: str,
    target_links: str | list[str] = DEFAULT_TARGET_LINKS,
    target_content: str | list[str] | None = None,
    valid_paths: str | list[str] | None = None,
    exclude_paths: str | list[str] | None = None,
    is_domain_match: bool | None = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: bool | None = DEFAULT_BASE_PATH_MATCH,
    is_links: bool | None = False,
) -> list[str]:
    # Claim the URL before fetching so concurrent workers never duplicate it
    with already_crawled_lock:
        if url in already_crawled:
            return []
        already_crawled.add(url)
    try:
        # Defined USER AGENTS for get URL
        headers_user_agents = {"User-Agent": random.choice(user_agents)}

        logger.debug("Crawling: %s", url)
        response = requests.get(url, headers=headers_user_agents, timeout=30)
        # Add a time sleep for avod rate limits
        sleep_time = random.uniform(1, 5)
        time.sleep(sleep_time)
    except requests.exceptions.RequestException:
        logger.exception("❌ Request error for %s", url)
        return []
    if response.status_code >= 400:
        logger.error("❌ HTTP %s for %s", response.status_code, url)
        return []
    content_type = (
        response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    )
    if content_type not in {"text/html", "application/xhtml+xml"}:
        logger.error("❌ Content not text/html for %s", url)
        return []

    # Use an explicit charset when the header does not provide one
    if not response.encoding:
        response.encoding = response.apparent_encoding

    # ---------------------------------
    # List of elements we want to strip
    # ---------------------------------
    strip_elements = []

    if is_links:
        strip_elements = ["a"]

    # -------------------------------
    # Create BS4 instance for parsing
    # -------------------------------
    soup = BeautifulSoup(response.text, "html.parser")

    # Strip unwanted tags
    for tag in soup(UNRENDERED_TAGS):
        tag.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not Path(file_path).exists():
        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------

        content = get_target_content(soup, target_content=target_content)

        if content:
            # --------------
            # Parse markdown
            # --------------
            try:
                # Flatten deeply nested content that would overflow markdownify's
                # recursion before conversion.
                if _max_nesting_depth(soup) > MAX_NESTING_DEPTH:
                    original_content = content
                    _flatten_deep_nesting(soup)
                    flattened_content = get_target_content(
                        soup, target_content=target_content
                    )
                    # The flatten may have removed the tags the selector matched;
                    # fall back to the pre-flatten content in that case.
                    content = (
                        flattened_content if flattened_content else original_content
                    )
                output = md(
                    content,
                    keep_inline_images_in=["td", "th", "a", "figure"],
                    strip=strip_elements,
                    escape_misc=True,
                )
            except RecursionError:
                logger.warning(
                    "❌ Content too deeply nested for %s, writing raw HTML.",
                    file_path,
                )
                output = content

            logger.info("Created 📝 %s", file_name)

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with Path(file_path).open("w", encoding="utf-8") as f:
                f.write(output)
        else:
            logger.error(
                "❌ Empty content for %s. Please check your targets skipping.",
                file_path,
            )

    child_urls = get_target_links(
        soup,
        base_url,
        target_links,
        valid_paths=valid_paths,
        exclude_paths=exclude_paths,
        is_domain_match=is_domain_match,
        is_base_path_match=is_base_path_match,
    )

    logger.debug("Found %s child URLs", len(child_urls) if child_urls else 0)
    return child_urls


def _max_nesting_depth(soup: BeautifulSoup) -> int:
    """Return the maximum ancestor depth of any tag in ``soup``."""
    max_depth = 0
    for tag in soup.find_all(True):
        depth = sum(1 for _ in tag.parents)
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _flatten_deep_nesting(soup: BeautifulSoup) -> None:
    """Collapse deeply nested container tags to stay under the recursion limit."""
    protected = {
        "a",
        "p",
        "pre",
        "code",
        "table",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "dl",
        "dt",
        "dd",
    }
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all(["div", "span", "section", "article", "main"]):
            if tag.name in protected:
                continue
            children = list(tag.find_all(True, recursive=False))
            if len(children) == 1 and children[0].name in {"div", "span", "section"}:
                tag.unwrap()
                changed = True
                break


def get_target_content(
    soup: BeautifulSoup, target_content: list[str] | None = None
) -> str:
    content = ""

    # Drop content the browser would never render
    for tag in soup(UNRENDERED_TAGS):
        tag.decompose()

    if target_content:
        seen: set[int] = set()
        for target in target_content:
            if not target or not target.strip():
                continue
            for tag in soup.select(target):
                # Skip tags already collected, nested inside one, or containing
                # one already collected (avoids duplication regardless of order)
                if (
                    id(tag) in seen
                    or any(id(parent) in seen for parent in tag.parents)
                    or any(id(descendant) in seen for descendant in tag.descendants)
                ):
                    continue
                seen.add(id(tag))
                content += str(tag)

    # ---------------------------
    # Naive estimation of content
    # ---------------------------
    else:
        max_text_length = 0
        ok = False
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag
                ok = True

        content = str(main_content) if ok else ""

    return content


def _path_includes(path: str, prefix: str) -> bool:
    """Check if ``path`` is ``prefix`` or starts at a segment boundary of it."""
    prefix_path = urllib.parse.urlparse(prefix).path.rstrip("/")
    if not prefix_path:
        return True
    return path == prefix_path or path.startswith(
        (prefix_path + "/", prefix_path + ":")
    )


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: list[str] = DEFAULT_TARGET_LINKS,
    valid_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    is_domain_match: bool | None = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: bool | None = DEFAULT_BASE_PATH_MATCH,
) -> list[str]:
    child_urls = []

    logger.info("valid_paths : %s", valid_paths)
    logger.info("exclude_paths : %s", exclude_paths)

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        child_urls.extend(
            urllib.parse.urljoin(base_url, link["href"].strip())
            for link in target.find_all("a")
            if link.get("href")
            and link["href"].strip()
            and not link["href"].lstrip().startswith("#")
        )

    base_hostname = urllib.parse.urlparse(base_url).hostname
    result = []
    for u in child_urls:
        child_url = urllib.parse.urlparse(u)

        # Drop non-http(s) schemes and fragment-only urls
        if child_url.scheme not in {"http", "https"}:
            continue

        # Drop invalid urls (e.g. leftover whitespace after urljoin)
        if not is_valid_url(u):
            continue

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if is_domain_match and child_url.hostname != base_hostname:
            continue

        if exclude_paths:
            excluded = False
            for exclude_path in exclude_paths:
                if _path_includes(child_url.path, exclude_path):
                    excluded = True
                    break
            if excluded:
                continue

        if is_base_path_match and _path_includes(
            child_url.path, urllib.parse.urlparse(base_url).path
        ):
            result.append(u)
            continue

        if valid_paths:
            for valid_path in valid_paths:
                if _path_includes(child_url.path, valid_path):
                    result.append(u)
                    break

    return result


# ------------------
# Worker thread logic
# ------------------
MAX_FILE_NAME_LENGTH = 200


def _unique_file_path(
    base_dir: str, file_name: str, used_files: dict[str, str], lock: threading.Lock
) -> str:
    base_path = f"{base_dir.rstrip('/') + '/'}{file_name}.md"
    with lock:
        if base_path not in used_files:
            used_files[base_path] = file_name
            return base_path
    # Collision: append an incrementing suffix until the path is free
    index = 1
    while True:
        candidate = f"{base_dir.rstrip('/') + '/'}{file_name}-{index}.md"
        with lock:
            if candidate not in used_files:
                used_files[candidate] = file_name
                return candidate
        index += 1


def worker(
    q: queue.Queue,
    base_url: str,
    max_depth: int,
    already_crawled: set,
    already_crawled_lock: threading.Lock,
    used_files: dict[str, str],
    base_dir: str,
    target_links: list[str] | None = DEFAULT_TARGET_LINKS,
    target_content: list[str] | None = None,
    valid_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    is_domain_match: bool | None = None,
    is_base_path_match: bool | None = None,
    is_links: bool | None = False,
) -> None:
    while True:
        item = q.get()
        try:
            if item is None:
                q.task_done()
                break
            depth, url = item
            if depth > max_depth:
                continue
            file_name = "-".join(re.findall(r"\w+", urllib.parse.urlparse(url).path))
            file_name = file_name if file_name else "index"
            file_name = file_name[:MAX_FILE_NAME_LENGTH]
            file_path = _unique_file_path(
                base_dir, file_name, used_files, already_crawled_lock
            )

            child_urls = crawl(
                url,
                base_url,
                already_crawled,
                already_crawled_lock,
                file_path,
                target_links,
                target_content,
                valid_paths,
                exclude_paths,
                is_domain_match,
                is_base_path_match,
                is_links,
            )
            child_urls = [normalize_url(u) for u in child_urls]
            for child_url in child_urls:
                q.put((depth + 1, child_url))
            time.sleep(1)
        except Exception:
            logger.exception("❌ Unexpected error while processing %s", item)
        finally:
            if item is not None:
                q.task_done()


# -----------------
# Thread management
# -----------------
def md_crawl(
    base_url: str,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    num_threads: int | None = DEFAULT_NUM_THREADS,
    base_dir: str | None = DEFAULT_BASE_DIR,
    target_links: str | list[str] = DEFAULT_TARGET_LINKS,
    target_content: str | list[str] | None = None,
    valid_paths: str | list[str] | None = None,
    exclude_paths: list[str] | None = None,
    is_domain_match: bool | None = None,
    is_base_path_match: bool | None = None,
    is_debug: bool | None = False,
    is_links: bool | None = False,
) -> None:
    if is_domain_match is False and is_base_path_match is True:
        raise ValueError("❌ Domain match must be True if base match is set to True")

    is_domain_match = (
        DEFAULT_DOMAIN_MATCH if is_domain_match is None else is_domain_match
    )
    is_base_path_match = (
        DEFAULT_BASE_PATH_MATCH if is_base_path_match is None else is_base_path_match
    )

    if not base_url:
        raise ValueError("❌ Base URL is required")

    if num_threads is None or num_threads < 1:
        raise ValueError("❌ num_threads must be at least 1")

    if max_depth is None or max_depth < 0:
        raise ValueError("❌ max_depth must be a non-negative integer")

    base_url = base_url.strip()

    if isinstance(target_links, str):
        target_links = (
            target_links.split(",") if "," in target_links else [target_links]
        )

    if isinstance(target_content, str):
        target_content = (
            target_content.split(",") if "," in target_content else [target_content]
        )

    if isinstance(valid_paths, str):
        valid_paths = valid_paths.split(",") if "," in valid_paths else [valid_paths]

    if isinstance(exclude_paths, str):
        exclude_paths = (
            exclude_paths.split(",") if "," in exclude_paths else [exclude_paths]
        )

    if is_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("🐞 Debugging enabled")
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(
        "🕸️ Crawling %s at ⏬ depth %s with 🧵 %s threads",
        base_url,
        max_depth,
        num_threads,
    )

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError("❌ Invalid base URL")

    # Normalize once so the queue, already_crawled and child links share one form
    base_url = normalize_url(base_url)

    # Create base_dir if it doesn't exist
    if not Path(base_dir).exists():
        Path(base_dir).mkdir(parents=True)

    already_crawled = set()
    already_crawled_lock = threading.Lock()
    used_files: dict[str, str] = {}

    # Create a queue of URLs to crawl
    q = queue.Queue()

    # Add the base URL to the queue
    q.put((0, base_url))

    threads = []

    # Create a thread for each URL in the queue
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                q,
                base_url,
                max_depth,
                already_crawled,
                already_crawled_lock,
                used_files,
                base_dir,
                target_links,
                target_content,
                valid_paths,
                exclude_paths,
                is_domain_match,
                is_base_path_match,
                is_links,
            ),
        )
        threads.append(t)
        t.start()
        logger.debug("Started thread %s of %s", i + 1, num_threads)

    # Wait for all queued URLs to be processed
    q.join()

    # Signal the workers to stop
    for _ in threads:
        q.put(None)

    # Wait for all threads to finish
    for t in threads:
        t.join()


logger.info("🏁 All threads have finished")
