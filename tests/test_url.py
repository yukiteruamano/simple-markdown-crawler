import pytest

from simple_markdown_crawler import is_valid_url, normalize_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/wiki/Page", True),
        ("http://example.com", True),
        ("ftp://example.com/file", True),
        ("not a url", False),
        ("", False),
        ("example.com", False),
        ("/wiki/relative", False),
        ("  https://example.com  ", False),
        ("  example.com  ", False),
        ("https://exa mple.com/x", False),
        ("https://example.com/x\n", False),
        ("https://example.com:99999/x", False),
        ("https://example.com:8080/x", True),
        ("https://user:pass@host.com/x", True),
    ],
)
def test_is_valid_url(url: str, expected: bool):
    assert is_valid_url(url) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/wiki/Page", "https://example.com/wiki/Page"),
        ("https://example.com/wiki/Page/", "https://example.com/wiki/Page"),
        ("https://example.com", "https://example.com"),
        ("https://example.com/?query=1#frag", "https://example.com?query=1"),
        ("https://example.com/a//b/", "https://example.com/a//b"),
        (
            "https://Example.com/wiki/A?lang=en",
            "https://example.com/wiki/A?lang=en",
        ),
    ],
)
def test_normalize_url(url: str, expected: str):
    assert normalize_url(url) == expected
