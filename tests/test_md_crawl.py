import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_markdown_crawler as crawler
from simple_markdown_crawler import md_crawl


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)


def _make_server(monkeypatch, pages):
    def fake_get(url, headers=None, timeout=None):
        html = pages.get(url)
        if html is None:
            raise crawler.requests.exceptions.ConnectionError(f"not found: {url}")
        return SimpleNamespace(
            text=html,
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)


def test_md_crawl_single_page(monkeypatch, tmp_path):
    pages = {"https://example.com/wiki/Page": '<div id="content"><h1>Title</h1></div>'}
    _make_server(monkeypatch, pages)
    md_crawl(
        "https://example.com/wiki/Page",
        base_dir=str(tmp_path),
        target_content=["div#content"],
        target_links=["body"],
    )
    md_file = tmp_path / "wiki-Page.md"
    assert md_file.exists()
    assert "Title" in md_file.read_text(encoding="utf-8")


def test_md_crawl_multiple_selectors_do_not_duplicate(monkeypatch, tmp_path):
    pages = {
        "https://example.com/wiki/Page": (
            '<div id="content"><article><p>Real content</p></article></div>'
        )
    }
    _make_server(monkeypatch, pages)
    md_crawl(
        "https://example.com/wiki/Page",
        base_dir=str(tmp_path),
        target_content=["div#content", "article", "p"],
        target_links=["body"],
    )
    md_file = tmp_path / "wiki-Page.md"
    content = md_file.read_text(encoding="utf-8")
    assert content.count("Real content") == 1


def test_md_crawl_follows_children(monkeypatch, tmp_path):
    pages = {
        "https://example.com/wiki/Root": (
            '<body><a href="/wiki/Child">Child</a></body>'
        ),
        "https://example.com/wiki/Child": "<body>child content</body>",
    }
    _make_server(monkeypatch, pages)
    md_crawl(
        "https://example.com/wiki/Root",
        max_depth=1,
        num_threads=2,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki"],
    )
    assert (tmp_path / "wiki-Root.md").exists()
    assert (tmp_path / "wiki-Child.md").exists()


def test_md_crawl_respects_max_depth(monkeypatch, tmp_path):
    pages = {
        "https://example.com/wiki/A": '<body><a href="/wiki/B">B</a></body>',
        "https://example.com/wiki/B": '<body><a href="/wiki/C">C</a></body>',
        "https://example.com/wiki/C": "<body>c</body>",
    }
    _make_server(monkeypatch, pages)
    md_crawl(
        "https://example.com/wiki/A",
        max_depth=1,
        num_threads=2,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki"],
    )
    assert (tmp_path / "wiki-A.md").exists()
    assert (tmp_path / "wiki-B.md").exists()
    assert not (tmp_path / "wiki-C.md").exists()


def test_md_crawl_does_not_duplicate_crawl(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return SimpleNamespace(
            text='<body><a href="/wiki/A">A</a></body>',
            headers={"Content-Type": "text/html"},
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    md_crawl(
        "https://example.com/wiki/A",
        max_depth=1,
        num_threads=3,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki"],
    )
    assert calls.count("https://example.com/wiki/A") == 1


def test_md_crawl_validates_base_url(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        md_crawl("not a url", base_dir=str(tmp_path))


def test_md_crawl_requires_base_url(tmp_path):
    with pytest.raises(ValueError):
        md_crawl("", base_dir=str(tmp_path))


def test_md_crawl_validates_num_threads(tmp_path):
    with pytest.raises(ValueError):
        md_crawl("https://example.com", num_threads=0, base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        md_crawl("https://example.com", num_threads=-1, base_dir=str(tmp_path))


def test_md_crawl_validates_max_depth(tmp_path):
    with pytest.raises(ValueError):
        md_crawl("https://example.com", max_depth=-1, base_dir=str(tmp_path))


def test_md_crawl_strips_base_url(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return SimpleNamespace(
            text="<body>hi</body>",
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    md_crawl(
        "  https://example.com/wiki/A  ",
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
    )
    assert calls == ["https://example.com/wiki/A"]


def test_md_crawl_normalizes_base_url_trailing_slash(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return SimpleNamespace(
            text='<body><a href="/wiki/Page">self</a>'
            '<a href="/wiki/Other">O</a></body>',
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    md_crawl(
        "https://example.com/wiki/Page/",
        max_depth=1,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki"],
    )
    assert calls == ["https://example.com/wiki/Page", "https://example.com/wiki/Other"]
    assert (tmp_path / "wiki-Page.md").exists()


def test_md_crawl_invalid_selector_does_not_hang(monkeypatch, tmp_path):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            text="<div>x</div>",
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        ),
    )
    md_crawl(
        "https://example.com",
        base_dir=str(tmp_path),
        target_content=["div["],
        num_threads=2,
    )


def test_md_crawl_resolves_filename_collision(monkeypatch, tmp_path):
    pages = {
        "https://example.com/a-b": '<body><a href="/a/b">B</a>hyphen</body>',
        "https://example.com/a/b": "<body>slash</body>",
    }
    _make_server(monkeypatch, pages)
    md_crawl(
        "https://example.com/a-b",
        max_depth=1,
        num_threads=2,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/a"],
    )
    md_files = sorted(p.name for p in tmp_path.glob("*.md"))
    assert len(md_files) == 2
    assert "a-b.md" in md_files
    assert "a-b-1.md" in md_files


def test_md_crawl_handles_long_filename(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return SimpleNamespace(
            text="<body>content</body>",
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    md_crawl(
        "https://example.com/wiki/" + "z" * 250,
        base_dir=str(tmp_path),
        target_content=["body"],
        target_links=["body"],
    )
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert len(md_files[0].name) < 255


def test_md_crawl_deeply_nested_page_not_lost(monkeypatch, tmp_path):
    deep_html = "<div>" * 500 + "<p>deep text</p>" + "</div>" * 500

    def fake_get(url, headers=None, timeout=None):
        return SimpleNamespace(
            text=deep_html,
            headers={"Content-Type": "text/html"},
            status_code=200,
            encoding="utf-8",
            apparent_encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    md_crawl(
        "https://example.com/wiki/A",
        base_dir=str(tmp_path),
        target_content=["div"],
        target_links=["body"],
    )
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert "deep text" in md_files[0].read_text(encoding="utf-8")


def test_unique_file_path_three_way_collision(tmp_path):
    used_files: dict[str, str] = {}
    lock = threading.Lock()
    paths = [
        crawler._unique_file_path(str(tmp_path), "a-b", used_files, lock)
        for _ in range(3)
    ]
    names = [Path(p).name for p in paths]
    assert names == ["a-b.md", "a-b-1.md", "a-b-2.md"]
    assert len(set(names)) == 3
