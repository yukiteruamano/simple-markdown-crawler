from bs4 import BeautifulSoup

from simple_markdown_crawler import get_target_content


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_get_target_content_with_selector():
    soup = _soup('<div id="content"><p>Hello</p></div><p>Ignored</p>')
    content = get_target_content(soup, target_content=["div#content"])
    assert "<p>Hello</p>" in content
    assert "Ignored" not in content


def test_get_target_content_with_multiple_selectors():
    soup = _soup("<article>A</article><main>B</main>")
    content = get_target_content(soup, target_content=["article", "main"])
    assert "A" in content
    assert "B" in content


def test_get_target_content_preserves_newlines():
    soup = _soup("<div><p>Line1</p>\n<p>Line2</p></div>")
    content = get_target_content(soup, target_content=["div"])
    assert "Line1" in content
    assert "Line2" in content


def test_get_target_content_preserves_pre_and_code():
    soup = _soup("<div><pre>a\nb\n  c</pre><code>x\ny</code></div>")
    content = get_target_content(soup, target_content=["div"])
    assert "a\nb\n  c" in content
    assert "x\ny" in content


def test_get_target_content_naive_largest_block():
    soup = _soup(
        "<html><body><p>short</p><div><h1>Title</h1>"
        "<p>longer body text here</p></div></body></html>"
    )
    content = get_target_content(soup)
    assert "longer body text here" in content


def test_get_target_content_returns_empty_string_when_empty():
    soup = _soup("<html><body></body></html>")
    assert get_target_content(soup) == ""


def test_get_target_content_selector_no_match_returns_empty():
    soup = _soup("<div>text</div>")
    assert get_target_content(soup, target_content=["#missing"]) == ""


def test_get_target_content_ignores_empty_selectors():
    soup = _soup("<div>hello</div>")
    assert get_target_content(soup, target_content=["div", "", "p"]) == (
        "<div>hello</div>"
    )
    assert get_target_content(soup, target_content=["  "]) == ""
    assert get_target_content(soup, target_content=[""]) == ""


def test_get_target_content_strips_unrendered_tags():
    soup = _soup("<div><template>TPL</template><iframe>IF</iframe><p>Real</p></div>")
    content = get_target_content(soup, target_content=["div"])
    assert "Real" in content
    assert "TPL" not in content
    assert "IF" not in content


def test_get_target_content_dedups_nested_selectors():
    soup = _soup('<div id="content"><article><p>inner</p></article></div>')
    content = get_target_content(soup, target_content=["div", "article", "p"])
    assert content.count("inner") == 1


def test_get_target_content_dedups_same_tag_via_selectors():
    soup = _soup('<div id="x">hello</div>')
    content = get_target_content(soup, target_content=["div", "#x"])
    assert content.count("hello") == 1


def test_get_target_content_keeps_distinct_siblings():
    soup = _soup("<div>A</div><span>B</span>")
    content = get_target_content(soup, target_content=["div", "span"])
    assert "A" in content
    assert "B" in content


def test_get_target_content_dedups_child_before_parent():
    soup = _soup(
        '<div id="bodyContent"><div class="mw-parser-output">'
        "<h1>Title</h1><p>Body text</p></div></div>"
    )
    content = get_target_content(
        soup, target_content=[".mw-parser-output", "#bodyContent"]
    )
    assert content.count("Title") == 1
    assert content.count("Body text") == 1


def test_get_target_content_naive_includes_table():
    soup = _soup(
        "<html><body><div><table><tr><th>Name</th><th>Desc</th></tr>"
        "<tr><td>Alice</td><td>Works on X</td></tr>"
        "</table></div><p>footer</p></body></html>"
    )
    content = get_target_content(soup)
    assert "<table>" in content


def test_get_target_content_naive_includes_pre():
    soup = _soup("<html><body><pre>code block here</pre></body></html>")
    content = get_target_content(soup)
    assert "<pre>" in content
