import threading
from types import SimpleNamespace

import pytest
import requests

import simple_markdown_crawler as crawler
from simple_markdown_crawler import crawl


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)


@pytest.fixture
def already_crawled():
    return set()


@pytest.fixture
def already_crawled_lock():
    return threading.Lock()


def _fake_response(
    text: str,
    content_type: str = "text/html; charset=utf-8",
    status: int = 200,
):
    return SimpleNamespace(
        text=text,
        headers={"Content-Type": content_type},
        status_code=status,
        encoding="utf-8",
        apparent_encoding="utf-8",
    )


def test_crawl_writes_markdown_file(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: _fake_response("<div><h1>Title</h1><p>Body</p></div>"),
    )
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/wiki/Page",
        "https://example.com/wiki/Page",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_links=["body"],
        target_content=["div"],
    )
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "Title" in content
    assert child_urls == []
    assert "https://example.com/wiki/Page" in already_crawled


def test_crawl_skips_already_crawled(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    already_crawled.add("https://example.com/wiki/Page")
    monkeypatch.setattr(
        crawler.requests, "get", lambda *a, **k: _fake_response("<div>x</div>")
    )
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/wiki/Page",
        "https://example.com/wiki/Page",
        already_crawled,
        already_crawled_lock,
        str(file_path),
    )
    assert child_urls == []
    assert not file_path.exists()


def test_crawl_skips_non_html(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: _fake_response("PNG", content_type="image/png"),
    )
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/img/x.png",
        "https://example.com",
        already_crawled,
        already_crawled_lock,
        str(file_path),
    )
    assert child_urls == []
    assert not file_path.exists()


def test_crawl_returns_empty_on_request_error(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(crawler.requests, "get", _raise)
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/wiki/Page",
        "https://example.com/wiki/Page",
        already_crawled,
        already_crawled_lock,
        str(file_path),
    )
    assert child_urls == []
    assert not file_path.exists()


def test_crawl_extracts_child_urls(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = '<body><div><a href="/wiki/A">A</a><a href="/wiki/B">B</a></div></body>'
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/wiki/",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_links=["body"],
    )
    assert child_urls == ["https://example.com/wiki/A", "https://example.com/wiki/B"]


def test_crawl_strips_links_when_is_links(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = '<div><a href="/wiki/A">link text</a></div>'
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/Page",
        "https://example.com/wiki/Page",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
        is_links=True,
    )
    content = file_path.read_text(encoding="utf-8")
    assert "link text" in content
    assert "[link text]" not in content


def test_crawl_does_not_overwrite_existing_file(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    file_path = tmp_path / "page.md"
    file_path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: _fake_response("<div>new content</div>"),
    )
    crawl(
        "https://example.com/wiki/Page",
        "https://example.com/wiki/Page",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    assert file_path.read_text(encoding="utf-8") == "existing"


def test_crawl_skips_http_error_status(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: _fake_response(
            "<html><h1>404 Not Found</h1></html>", status=404
        ),
    )
    file_path = tmp_path / "page.md"
    child_urls = crawl(
        "https://example.com/missing",
        "https://example.com",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["body"],
    )
    assert child_urls == []
    assert not file_path.exists()


def test_crawl_accepts_xhtml_content_type(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *a, **k: _fake_response(
            "<body><div>x</div></body>", content_type="application/xhtml+xml"
        ),
    )
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/xhtml",
        "https://example.com",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    assert file_path.exists()


def test_crawl_strips_unrendered_tags(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = (
        "<body><div><template>SECRET</template><iframe>IF</iframe>"
        "<noscript>NS</noscript><svg>SVG</svg><p>Real content</p></div></body>"
    )
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["body"],
    )
    content = file_path.read_text(encoding="utf-8")
    assert "Real content" in content
    for leak in ("SECRET", "IF", "NS", "SVG"):
        assert leak not in content


def test_crawl_handles_deeply_nested_html(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = "<div>" * 500 + "<p>deep text</p>" + "</div>" * 500
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "deep text" in content
    assert "<div>" not in content


def test_crawl_very_deep_nesting_writes_fallback(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = "<div>" * 1200 + "<p>deep text</p>" + "</div>" * 1200
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    assert file_path.exists()
    assert "deep text" in file_path.read_text(encoding="utf-8")


def test_crawl_mixed_deep_nesting_not_empty(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = (
        "<div>" * 200
        + "<span>" * 200
        + "<section>" * 200
        + "<article>" * 200
        + "<p>deep text here</p>"
        + "</article>" * 200
        + "</section>" * 200
        + "</span>" * 200
        + "</div>" * 200
    )
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "deep text here" in content
    assert len(content) > 0


def test_crawl_escapes_misc_markdown_chars(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = (
        "<div><p>Intro</p><p>1. First point</p>"
        "<p>a | b | c</p><p># not heading</p></div>"
    )
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    content = file_path.read_text(encoding="utf-8")
    assert r"1\. First point" in content
    assert r"a \| b \| c" in content
    assert r"\# not heading" in content


def test_crawl_does_not_escape_inside_code_blocks(
    monkeypatch, tmp_path, already_crawled, already_crawled_lock
):
    html = "<div><pre>*not bold* _not emph_</pre></div>"
    monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: _fake_response(html))
    file_path = tmp_path / "page.md"
    crawl(
        "https://example.com/wiki/A",
        "https://example.com/wiki/",
        already_crawled,
        already_crawled_lock,
        str(file_path),
        target_content=["div"],
    )
    content = file_path.read_text(encoding="utf-8")
    assert "*not bold* _not emph_" in content
