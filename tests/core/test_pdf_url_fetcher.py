"""Tests for the WeasyPrint URL fetcher SSRF guard in intentkit.utils.pdf."""

import socket
from collections.abc import Iterator
from unittest.mock import MagicMock, patch
from urllib import request

import pytest
from langchain_core.tools.base import ToolException
from weasyprint.urls import URLFetcher, URLFetcherResponse

from intentkit.utils.pdf import _POST_TEMPLATE, _make_url_fetcher, _render_template

_GETADDRINFO = "intentkit.utils.ssrf.socket.getaddrinfo"
_PUBLIC_URL = "https://cdn.example.com/cover.png"


def _addrinfo(ip: str):
    """Build a minimal getaddrinfo-style result for a single address."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


@pytest.fixture
def base_fetch() -> Iterator[MagicMock]:
    """Stub WeasyPrint's own fetch so no test opens a socket."""
    with patch("weasyprint.urls.URLFetcher.fetch") as mock:
        yield mock


@pytest.fixture
def fetcher() -> URLFetcher:
    return _make_url_fetcher()


@pytest.mark.parametrize(
    ("url", "resolve"),
    [
        ("file:///etc/passwd", {}),
        ("http://localhost/x.png", {"return_value": _addrinfo("127.0.0.1")}),
        (
            "http://169.254.169.254/latest/meta-data/",
            {"return_value": _addrinfo("169.254.169.254")},
        ),
        ("http://internal.svc/logo.png", {"return_value": _addrinfo("10.1.2.3")}),
        ("http://nope.invalid/x.png", {"side_effect": socket.gaierror}),
    ],
    ids=["non-http-scheme", "loopback", "metadata", "private", "unresolvable"],
)
def test_blocked_url_raises_before_fetching(
    fetcher: URLFetcher, base_fetch: MagicMock, url: str, resolve: dict
):
    """A refused URL raises out of ``fetch`` — WeasyPrint turns that into a
    logged, non-fatal ``URLFetchingError`` — and never reaches the real fetch.
    The unresolvable case fails closed; the scheme case needs no lookup."""
    with patch(_GETADDRINFO, **resolve), pytest.raises(ToolException):
        fetcher.fetch(url)
    base_fetch.assert_not_called()


def test_blocked_redirect_target_leaves_no_stale_request(fetcher: URLFetcher):
    """urllib's redirect handler re-enters ``open`` with a ``Request`` for the
    new location, which the base class stashes on the instance for ``fetch``
    to consume. The guard must fire on that path and must not leave the stash
    behind, or the next fetch would open the blocked target."""
    with (
        patch(_GETADDRINFO, return_value=_addrinfo("10.1.2.3")),
        pytest.raises(ToolException),
    ):
        fetcher.open(request.Request("http://internal.svc/after-302"))

    opened = MagicMock(url=_PUBLIC_URL, status=200, headers={})
    with (
        patch(_GETADDRINFO, return_value=_addrinfo("93.184.216.34")),
        patch("urllib.request.OpenerDirector.open", return_value=opened) as open_mock,
    ):
        fetcher.fetch(_PUBLIC_URL)
    assert open_mock.call_args.args[0].full_url == _PUBLIC_URL


@pytest.mark.usefixtures("stub_public_dns")
def test_allows_public_host(fetcher: URLFetcher, base_fetch: MagicMock):
    """A public address is allowed and delegates to WeasyPrint's fetcher."""
    sentinel = URLFetcherResponse(_PUBLIC_URL, b"img")
    base_fetch.return_value = sentinel

    assert fetcher.fetch(_PUBLIC_URL) is sentinel
    base_fetch.assert_called_once()


def test_rendered_post_html_has_no_cover():
    """The exported PDF must not render a post cover image.

    Covers are list-only; even if a stray ``cover`` value is passed through, the
    template should ignore it and emit no cover ``<img>``.
    """
    html = _render_template(
        _POST_TEMPLATE,
        title="Hello World",
        content="<p>Body content</p>",
        agent_name="Agent",
        date="January 01, 2026",
        tags=["alpha"],
        cover="https://cdn.example.com/cover.png",
    )

    assert "post-cover" not in html
    assert "cover.png" not in html
    # The rest of the post still renders.
    assert "Hello World" in html
    assert "<p>Body content</p>" in html
    assert "alpha" in html
