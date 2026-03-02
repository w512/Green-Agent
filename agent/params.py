"""Validation helpers for tool arguments (values come from model JSON)."""

from __future__ import annotations

import math


def require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    return value


def optional_int(value: object, name: str, default: int) -> int:
    """Accept JSON numbers (int or float), truncating toward zero."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return math.trunc(value)


def positive_int(value: object, name: str, default: int) -> int:
    result = optional_int(value, name, default)
    if result < 1:
        raise ValueError(f"{name} must be a positive number.")
    return result
