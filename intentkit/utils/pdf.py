"""PDF generation utilities for post content."""

from __future__ import annotations

import asyncio
import re
import ssl
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi.responses import Response
from langchain_core.tools.base import ToolException

from intentkit.utils.ssrf import validate_fetch_url_sync

if TYPE_CHECKING:
    from weasyprint.urls import URLFetcher, URLFetcherResponse

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_POST_TEMPLATE = (_TEMPLATE_DIR / "post_pdf.html").read_text()

_MD_EXTENSIONS = [
    "fenced_code",
    "tables",
    "codehilite",
    "pymdownx.tasklist",
]

_MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "css_class": "codehilite",
        "guess_lang": False,
    },
    "pymdownx.tasklist": {
        "custom_checkbox": True,
    },
}

_FOR_PATTERN = re.compile(
    r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}",
    re.DOTALL,
)
_IF_PATTERN = re.compile(
    r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}",
    re.DOTALL,
)
_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Variables that contain pre-rendered HTML and should not be escaped
_RAW_VARS = {"content"}


# URLFetcher builds an HTTPS handler per instance and, given no context, has
# ssl load the system trust store each time (~4ms). One context serves every
# render; the fetcher instance itself must stay per-render, it carries
# per-request state.
_SSL_CONTEXT = ssl.create_default_context()


def _make_url_fetcher() -> URLFetcher:
    """Build the WeasyPrint URL fetcher that blocks non-HTTP schemes and
    non-public hosts (SSRF prevention).

    WeasyPrint 70 replaced the fetcher callable with a ``URLFetcher`` class
    (a ``urllib`` opener), so the guard is a subclass. A refused asset raises
    out of ``fetch``; WeasyPrint wraps that into a non-fatal
    ``URLFetchingError`` (its ``fail_on_errors`` default), logs the reason and
    renders on without the asset. Redirects are covered too: the opener's
    redirect handler re-enters ``open`` with the new URL, which routes back
    through ``fetch`` and hence the guard.

    Imported lazily: ``weasyprint`` needs the native pango/cairo libraries,
    which only the app image installs.
    """
    from weasyprint.urls import URLFetcher

    class SafeURLFetcher(URLFetcher):
        def fetch(
            self, url: str, headers: dict[str, str] | None = None
        ) -> URLFetcherResponse:
            try:
                validate_fetch_url_sync(url)
            except ToolException:
                # A redirect arrives via open(), which stashes its Request on
                # self for the base fetch to consume and clear. Refusing here
                # skips that clear, and the next fetch would pick the blocked
                # target up instead of its own URL.
                self._request = None
                raise
            return super().fetch(url, headers=headers)

    return SafeURLFetcher(ssl_context=_SSL_CONTEXT)


def _render_template(template: str, **kwargs: object) -> str:
    """Simple template rendering with {{ var }}, {% if %}, {% for %} support."""
    result = template

    for match in _FOR_PATTERN.finditer(result):
        var_name = match.group(1)
        list_name = match.group(2)
        body = match.group(3)
        items = kwargs.get(list_name, [])
        rendered = ""
        if items and hasattr(items, "__iter__"):
            for item in items:  # pyright: ignore[reportGeneralTypeIssues]
                rendered += body.replace("{{ " + var_name + " }}", escape(str(item)))
        result = result.replace(match.group(0), rendered, 1)

    for match in _IF_PATTERN.finditer(result):
        var_name = match.group(1)
        body = match.group(2)
        value = kwargs.get(var_name)
        if value:
            result = result.replace(match.group(0), body, 1)
        else:
            result = result.replace(match.group(0), "", 1)

    for match in _VAR_PATTERN.finditer(result):
        var_name = match.group(1)
        value = kwargs.get(var_name, "")
        safe_value = str(value) if var_name in _RAW_VARS else escape(str(value))
        result = result.replace(match.group(0), safe_value, 1)

    return result


def _generate_pdf(
    title: str,
    markdown_content: str,
    agent_name: str,
    created_at: datetime,
    tags: list[str] | None = None,
) -> bytes:
    """Convert post content to a styled PDF. Runs synchronously (CPU-bound)."""
    import markdown as md
    from weasyprint import HTML

    html_body = md.markdown(
        markdown_content,
        extensions=_MD_EXTENSIONS,
        extension_configs=_MD_EXTENSION_CONFIGS,
    )

    date_str = created_at.strftime("%B %d, %Y")

    full_html = _render_template(
        _POST_TEMPLATE,
        title=title,
        content=html_body,
        agent_name=agent_name,
        date=date_str,
        tags=tags or [],
    )

    result = HTML(string=full_html, url_fetcher=_make_url_fetcher()).write_pdf()
    if result is None:
        raise RuntimeError("WeasyPrint failed to generate PDF")
    return result


async def generate_post_pdf(
    title: str,
    markdown_content: str,
    agent_name: str,
    created_at: datetime,
    tags: list[str] | None = None,
) -> bytes:
    """Convert post content to a styled PDF asynchronously.

    Wraps the CPU-bound WeasyPrint rendering in a thread to avoid
    blocking the event loop.
    """
    return await asyncio.to_thread(
        _generate_pdf,
        title,
        markdown_content,
        agent_name,
        created_at,
        tags,
    )


async def post_pdf_response(
    post: Any,
    filename: str | None = None,
) -> Response:
    """Generate a PDF from a post record and return it as a download Response.

    Accepts any object with title, markdown, agent_name, created_at, tags,
    slug, and id attributes (e.g., an enriched AgentPost).
    """
    pdf_bytes = await generate_post_pdf(
        title=post.title,
        markdown_content=post.markdown,
        agent_name=post.agent_name or post.agent_id,
        created_at=post.created_at,
        tags=post.tags,
    )
    fname = filename or f"{post.slug or post.id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
