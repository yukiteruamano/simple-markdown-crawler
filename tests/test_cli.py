import pytest

import simple_markdown_crawler.cli as cli
from simple_markdown_crawler import (
    DEFAULT_BASE_DIR,
    DEFAULT_MAX_DEPTH,
    DEFAULT_NUM_THREADS,
)


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_md_crawl(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "md_crawl", fake_md_crawl)
    return calls


@pytest.fixture
def run_cli(monkeypatch):
    def _run(argv):
        monkeypatch.setattr("sys.argv", argv)
        cli.main()

    return _run


def test_cli_passes_base_url_and_defaults(run_cli, captured):
    run_cli(["simple-markdown-crawler", "https://example.com/wiki/Page"])
    kwargs = captured["kwargs"]
    assert captured["args"] == ("https://example.com/wiki/Page",)
    assert kwargs["max_depth"] == DEFAULT_MAX_DEPTH
    assert kwargs["num_threads"] == DEFAULT_NUM_THREADS
    assert kwargs["base_dir"] == DEFAULT_BASE_DIR
    assert kwargs["target_links"] is None
    assert kwargs["target_content"] is None
    assert kwargs["valid_paths"] is None
    assert kwargs["exclude_paths"] is None
    assert kwargs["is_links"] is False


def test_cli_parses_lists(run_cli, captured):
    run_cli(
        [
            "simple-markdown-crawler",
            "--target-content",
            "div#content,article",
            "--target-links",
            "body",
            "--valid-paths",
            "/wiki,/help",
            "--exclude-paths",
            "/admin",
            "https://example.com/wiki/Page",
        ]
    )
    kwargs = captured["kwargs"]
    assert kwargs["target_content"] == ["div#content", "article"]
    assert kwargs["target_links"] == ["body"]
    assert kwargs["valid_paths"] == ["/wiki", "/help"]
    assert kwargs["exclude_paths"] == ["/admin"]


def test_cli_defaults_do_not_produce_nested_lists(run_cli, captured):
    run_cli(["simple-markdown-crawler", "https://example.com/wiki/Page"])
    assert captured["kwargs"]["target_links"] is None
    assert captured["kwargs"]["target_content"] is None


def test_cli_links_flag_toggles(run_cli, captured):
    run_cli(
        [
            "simple-markdown-crawler",
            "--links",
            "https://example.com/wiki/Page",
        ]
    )
    assert captured["kwargs"]["is_links"] is True


def test_cli_flags(run_cli, captured):
    run_cli(
        [
            "simple-markdown-crawler",
            "-d",
            "3",
            "-t",
            "5",
            "-b",
            "./out",
            "-e",
            "https://example.com/wiki/Page",
        ]
    )
    kwargs = captured["kwargs"]
    assert kwargs["max_depth"] == 3
    assert kwargs["num_threads"] == 5
    assert kwargs["base_dir"] == "./out"
    assert kwargs["is_debug"] is True


def test_cli_domain_match_toggle(run_cli, captured):
    run_cli(
        [
            "simple-markdown-crawler",
            "--no-domain-match",
            "https://example.com/wiki/Page",
        ]
    )
    assert captured["kwargs"]["is_domain_match"] is False

    run_cli(
        [
            "simple-markdown-crawler",
            "--domain-match",
            "https://example.com/wiki/Page",
        ]
    )
    assert captured["kwargs"]["is_domain_match"] is True


def test_cli_base_path_match_toggle(run_cli, captured):
    run_cli(
        [
            "simple-markdown-crawler",
            "--no-base-path-match",
            "https://example.com/wiki/Page",
        ]
    )
    assert captured["kwargs"]["is_base_path_match"] is False

    run_cli(
        [
            "simple-markdown-crawler",
            "--base-path-match",
            "https://example.com/wiki/Page",
        ]
    )
    assert captured["kwargs"]["is_base_path_match"] is True


def test_cli_catches_md_crawl_value_error(monkeypatch, capsys):
    def raising(*args, **kwargs):
        raise ValueError("❌ num_threads must be at least 1")

    monkeypatch.setattr(cli, "md_crawl", raising)
    monkeypatch.setattr(
        "sys.argv",
        ["simple-markdown-crawler", "-t", "0", "https://example.com/wiki/Page"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "num_threads" in err
    assert "Traceback" not in err


def test_cli_help_does_not_print_banner(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["simple-markdown-crawler", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "crawler that recursively" not in out
