"""fetch: HTTP GET with HTML-to-text conversion."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from agent.params import positive_int, require_text

if TYPE_CHECKING:
    from agent.tools import Environment

DEFAULT_MAX = 30_000
HARD_MAX = 60_000
TIMEOUT_SECONDS = 15.0
MAX_BODY_BYTES = 2 * 1024 * 1024
USER_AGENT = "pyagent/0.1 (+https://github.com/; coding agent)"
ACCEPT = "text/html, text/plain, application/json;q=0.9, */*;q=0.5"

TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
)
TEXT_SUFFIXES = ("+json", "+xml")


def assert_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only http: and https: URLs are allowed.")
    return url


def is_text_type(content_type: str) -> bool:
    return content_type.startswith(TEXT_TYPES) or content_type.endswith(
        TEXT_SUFFIXES
    )


# --- HTML to text -----------------------------------------------------------

SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "li", "ul", "ol", "table", "tr", "section",
        "article", "header", "footer", "nav", "aside", "main", "pre",
        "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd",
        "figure", "figcaption", "details", "summary",
    }
)  # fmt: skip


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_pre = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "pre":
            self._in_pre += 1
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "pre":
            self._in_pre = max(0, self._in_pre - 1)
            self.parts.append("\n")
        elif tag in BLOCK_TAGS and tag != "li":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_pre:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [line.rstrip() for line in raw.split("\n")]
        collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return collapsed.strip()


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


# --- tool -------------------------------------------------------------------


TIMEOUT_MESSAGE = f"fetch exceeded the {TIMEOUT_SECONDS:g} second timeout."


def _open(request: urllib.request.Request) -> Any:
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        return error  # 4xx/5xx still carry a body worth returning
    except TimeoutError as error:
        raise ValueError(TIMEOUT_MESSAGE) from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, TimeoutError):
            raise ValueError(TIMEOUT_MESSAGE) from error
        raise ValueError(f"fetch failed: {reason}") from error


def fetch_url(url: str, max_chars: int) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": ACCEPT}
    request = urllib.request.Request(url, headers=headers)

    with _open(request) as response:
        final_url = assert_http_url(response.geturl() or url)
        status = getattr(response, "status", None) or response.getcode()
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        if not is_text_type(content_type):
            raise ValueError(f"Unsupported content type: {content_type}")
        try:
            raw = response.read(MAX_BODY_BYTES)
        except TimeoutError as error:
            raise ValueError(TIMEOUT_MESSAGE) from error

    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")
    if content_type == "text/html":
        body = html_to_text(body)

    header = (
        f"status: {status}\nurl: {final_url}\ncontent-type: {content_type}\n\n"
    )
    if len(body) <= max_chars:
        return header + body
    note = f"\n...[body truncated at {max_chars} chars]"
    return f"{header}{body[:max_chars]}{note}"


class FetchTool:
    needs_approval = True
    trust = "always"

    def __init__(self, env: Environment) -> None:
        pass

    def describe(self, args: dict[str, Any]) -> str:
        return f"fetch {args.get('url')}"

    def execute(self, args: dict[str, Any]) -> str:
        url = assert_http_url(require_text(args.get("url"), "url"))
        max_chars = min(
            positive_int(args.get("max_chars"), "max_chars", DEFAULT_MAX),
            HARD_MAX,
        )
        return fetch_url(url, max_chars)


def create_tool(env: Environment) -> FetchTool:
    return FetchTool(env)
