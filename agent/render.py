"""Compact, one-glance text views of agent events, shared by the frontends."""

from __future__ import annotations

import json

PREVIEW_LINES = 8
PREVIEW_CHARS = 600
ARG_VALUE_CHARS = 60
ARGS_CHARS = 200


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_args(args: object, args_text: object = None) -> str:
    """One-line view of tool arguments with long values trimmed."""
    if not isinstance(args, dict):
        return shorten(str(args_text or ""), ARGS_CHARS)
    trimmed = {
        key: shorten(value, ARG_VALUE_CHARS)
        if isinstance(value, str)
        else value
        for key, value in args.items()
    }
    return shorten(json.dumps(trimmed, ensure_ascii=False), ARGS_CHARS)


def preview(text: str) -> str:
    """First lines of a tool result, with a note when more was omitted."""
    lines = text.splitlines() or [""]
    shown = lines[:PREVIEW_LINES]
    body = "\n".join(shown)[:PREVIEW_CHARS]
    if len(lines) > len(shown) or len(body) < len(text):
        body += f"\n... ({len(lines)} lines total)"
    return body
