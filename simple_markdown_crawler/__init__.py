from bs4 import BeautifulSoup
import urllib.parse
import threading
from markdownify import markdownify as md
import requests
import logging
import queue
import time
import os
import random
import re
from typing import List, Optional, Union

__version__ = "0.1.0"
__author__ = "Jose Maldonado (github.com/yukiteruamano)"
__copyright__ = "(C) 2023 Paul Pierre. MIT License. 2025 Jose Maldonado. MIT License"
__contributors__ = ["Jose Maldonado"]

BANNER = """
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
DEFAULT_TARGET_CONTENT = ["article", "div", "main", "p"]
DEFAULT_TARGET_LINKS = ["body"]
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True


# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        logger.debug(f"❌ Invalid URL {url}")
        return False


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), None, None, None)
    )


# ------------------
# HTML parsing logic
# ------------------

# User Agents List
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/96.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Vivaldi/6.5.3267.48",
]


def crawl(
    url: str,
    base_url: str,
    already_crawled: set,
    file_path: str,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    exclude_paths: Union[str, List[str]] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
    is_links: Optional[bool] = False,
) -> List[str]:
    if url in already_crawled:
        return []
    try:
        # Defined USER AGENTS for get URL
        headers_user_agents = {"User-Agent": random.choice(user_agents)}

        logger.debug(f"Crawling: {url}")
        response = requests.get(url, headers=headers_user_agents)
        # Add a time sleep for avod rate limits
        sleep_time = random.uniform(1, 5)
        time.sleep(sleep_time)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error for {url}: {e}")
        return []
    if "text/html" not in response.headers.get("Content-Type", ""):
        logger.error(f"❌ Content not text/html for {url}")
        return []
    already_crawled.add(url)

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
    for script in soup(["script", "style"]):
        script.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not os.path.exists(file_path):
        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------

        content = get_target_content(soup, target_content=target_content)

        if content:
            # logger.error(f'❌ Empty content for {file_path}. Please check your targets skipping.')
            # return []

            # --------------
            # Parse markdown
            # --------------
            output = md(
                content,
                keep_inline_images_in=["td", "th", "a", "figure"],
                strip=strip_elements,
            )

            logger.info(f"Created 📝 {file_name}")

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            logger.error(
                f"❌ Empty content for {file_path}. Please check your targets skipping."
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

    logger.debug(f"Found {len(child_urls) if child_urls else 0} child URLs")
    return child_urls


def get_target_content(
    soup: BeautifulSoup, target_content: Union[List[str], None] = None
) -> str:
    content = ""

    # -------------------------------------
    # Get target content by target selector
    # -------------------------------------
    if target_content:
        for target in target_content:
            for tag in soup.select(target):
                content += f"{str(tag)}".replace("\n", "")

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

    return content if len(content) > 0 else False


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: List[str] = DEFAULT_TARGET_LINKS,
    valid_paths: Union[List[str], None] = None,
    exclude_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
) -> List[str]:
    child_urls = []

    logger.info(f"valid_paths : {valid_paths}")
    logger.info(f"exclude_paths : {exclude_paths}")

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        for link in target.find_all("a"):
            child_urls.append(urllib.parse.urljoin(base_url, link.get("href")))

    result = []
    for u in child_urls:
        child_url = urllib.parse.urlparse(u)

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if (
            is_domain_match
            and child_url.netloc != urllib.parse.urlparse(base_url).netloc
        ):
            continue

        if exclude_paths:
            excluded = False
            for exclude_path in exclude_paths:
                if child_url.path.startswith(urllib.parse.urlparse(exclude_path).path):
                    excluded = True
                    break
            if excluded:
                continue

        if is_base_path_match and child_url.path.startswith(
            urllib.parse.urlparse(base_url).path
        ):
            result.append(u)
            continue

        if valid_paths:
            for valid_path in valid_paths:
                if child_url.path.startswith(urllib.parse.urlparse(valid_path).path):
                    result.append(u)
                    break

    return result


# ------------------
# Worker thread logic
# ------------------
def worker(
    q: object,
    base_url: str,
    max_depth: int,
    already_crawled: set,
    base_dir: str,
    target_links: Union[List[str], None] = DEFAULT_TARGET_LINKS,
    target_content: Union[List[str], None] = None,
    valid_paths: Union[List[str], None] = None,
    exclude_paths: Union[List[str], None] = None,
    is_domain_match: bool = None,
    is_base_path_match: bool = None,
    is_links: Optional[bool] = False,
) -> None:
    while not q.empty():
        depth, url = q.get()
        if depth > max_depth:
            continue
        file_name = "-".join(re.findall(r"\w+", urllib.parse.urlparse(url).path))
        file_name = "index" if not file_name else file_name
        file_path = f"{base_dir.rstrip('/') + '/'}{file_name}.md"

        child_urls = crawl(
            url,
            base_url,
            already_crawled,
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


# -----------------
# Thread management
# -----------------
def md_crawl(
    base_url: str,
    max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
    num_threads: Optional[int] = DEFAULT_NUM_THREADS,
    base_dir: Optional[str] = DEFAULT_BASE_DIR,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    exclude_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = None,
    is_base_path_match: Optional[bool] = None,
    is_debug: Optional[bool] = False,
    is_links: Optional[bool] = False,
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
        f"🕸️ Crawling {base_url} at ⏬ depth {max_depth} with 🧵 {num_threads} threads"
    )

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError("❌ Invalid base URL")

    # Create base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    already_crawled = set()

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
        logger.debug(f"Started thread {i + 1} of {num_threads}")

    # Wait for all threads to finish
    for t in threads:
        t.join()


logger.info("🏁 All threads have finished")
