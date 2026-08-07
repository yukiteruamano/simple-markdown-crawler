from bs4 import BeautifulSoup

from simple_markdown_crawler import get_target_links

BASE_URL = "https://example.com/wiki/"


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_get_target_links_same_domain():
    soup = _soup(
        '<body><a href="/wiki/A">A</a><a href="https://external.com/x">X</a></body>'
    )
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_domain_match_disabled():
    soup = _soup(
        '<body><a href="/wiki/A">A</a><a href="https://external.com/wiki/X">X</a></body>'
    )
    links = get_target_links(
        soup,
        BASE_URL,
        target_links=["body"],
        is_domain_match=False,
    )
    assert "https://external.com/wiki/X" in links


def test_get_target_links_base_path_match():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/other/B">B</a></body>')
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_valid_paths():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/cats/B">B</a></body>')
    links = get_target_links(
        soup,
        BASE_URL,
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki", "/cats"],
    )
    assert links == ["https://example.com/wiki/A", "https://example.com/cats/B"]


def test_get_target_links_exclude_paths():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/wiki/Special:B">S</a></body>')
    links = get_target_links(
        soup,
        BASE_URL,
        target_links=["body"],
        exclude_paths=["/wiki/Special"],
    )
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_target_scoped():
    soup = _soup(
        '<nav><a href="/nav/A">A</a></nav><body><a href="/body/B">B</a></body>'
    )
    links = get_target_links(soup, "https://example.com/", target_links=["nav"])
    assert links == ["https://example.com/nav/A"]


def test_get_target_links_nested_list_regression():
    soup = _soup('<body><a href="/wiki/A">A</a></body>')
    links = get_target_links(soup, BASE_URL, target_links=[["body"]])
    assert links == []


def test_get_target_links_relative_url_join():
    soup = _soup('<body><a href="Sub">Sub</a></body>')
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/Sub"]


def test_get_target_links_ignores_empty_and_fragment_hrefs():
    soup = _soup(
        '<body><a href="">empty</a><a>missing</a><a href="#">hash</a>'
        '<a href="#section">sec</a><a href="/wiki/A">A</a></body>'
    )
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_ignores_non_http_schemes():
    soup = _soup(
        '<body><a href="javascript:void(0)">js</a>'
        '<a href="mailto:a@b.com">mail</a><a href="tel:+1">tel</a>'
        '<a href="/wiki/A">A</a></body>'
    )
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_domain_match_case_insensitive():
    soup = _soup('<body><a href="https://EXAMPLE.com/wiki/A">A</a></body>')
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://EXAMPLE.com/wiki/A"]


def test_get_target_links_valid_paths_segment_boundary():
    soup = _soup(
        '<body><a href="/wiki">W</a><a href="/wikispecial">WS</a>'
        '<a href="/wiki/A">A</a></body>'
    )
    links = get_target_links(
        soup,
        "https://example.com/",
        target_links=["body"],
        is_base_path_match=False,
        valid_paths=["/wiki"],
    )
    assert "https://example.com/wikispecial" not in links
    assert "https://example.com/wiki" in links
    assert "https://example.com/wiki/A" in links


def test_get_target_links_exclude_paths_segment_boundary():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/wikispecial">WS</a></body>')
    links = get_target_links(
        soup,
        "https://example.com/",
        target_links=["body"],
        exclude_paths=["/wiki"],
    )
    assert links == ["https://example.com/wikispecial"]


def test_get_target_links_exclude_subpage_colon():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/wiki/Special:B">S</a></body>')
    links = get_target_links(
        soup,
        BASE_URL,
        target_links=["body"],
        exclude_paths=["/wiki/Special"],
    )
    assert links == ["https://example.com/wiki/A"]


def test_get_target_links_strips_href_whitespace():
    soup = _soup('<body><a href=" /wiki/A ">A</a><a href="/wiki/B">B</a></body>')
    links = get_target_links(soup, BASE_URL, target_links=["body"])
    assert links == ["https://example.com/wiki/A", "https://example.com/wiki/B"]


def test_get_target_links_base_path_match_with_query():
    soup = _soup('<body><a href="/wiki/A">A</a><a href="/other">O</a></body>')
    links = get_target_links(
        soup, "https://example.com/wiki/Page?x=1", target_links=["body"]
    )
    assert links == []
