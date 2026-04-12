"""OpenAI-compatible Chat Completions client with retry and model fallback.

Retry policy (per model): up to RETRY_ATTEMPTS on 429/503 or when the
error text carries a known "busy" marker, with linear backoff. On 503, or
when attempts run out, switch to the next model in the fallback list.
The provider remembers the model that answered and uses it next time.

The SDK's own retries are disabled so this policy is the only one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from openai import OpenAI

RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0
RETRY_STATUSES = frozenset({429, 503})
RETRY_MARKERS = ("UNAVAILABLE", "high demand")

Log = Callable[[str], None]
Sleep = Callable[[float], None]


class ConfigError(Exception):
    """A required setting is missing."""


def status_of(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(error: BaseException) -> bool:
    if status_of(error) in RETRY_STATUSES:
        return True
    text = str(error)
    return any(marker in text for marker in RETRY_MARKERS)


def unique(values: Iterable[object]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in seen:
            seen.append(value)
    return seen


def require_setting(value: object, name: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if text:
        return text
    raise ConfigError(f"{name} is not set. Set it in config.py.")


def create_with_retry(
    client: Any,
    body: dict[str, Any],
    models: Sequence[str],
    log: Log,
    sleep: Sleep = time.sleep,
) -> tuple[Any, str]:
    """Call chat.completions.create; return (response, model that answered)."""
    last_error: BaseException | None = None

    for index, model in enumerate(models):
        next_model = models[index + 1] if index + 1 < len(models) else None
        request = {**body, "model": model}

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = client.chat.completions.create(**request)
                return response, model
            except Exception as error:
                last_error = error
                if not is_retryable(error):
                    raise
                status = status_of(error) or 503
                last_attempt = attempt == RETRY_ATTEMPTS
                if next_model and (last_attempt or status == 503):
                    switch = f"{model} -> {next_model}"
                    log(f"model busy ({status}); switching {switch}")
                    break
                if last_attempt:
                    raise
                wait = RETRY_DELAY_SECONDS * attempt
                retry = f"{attempt}/{RETRY_ATTEMPTS - 1}"
                log(f"model busy ({status}); retry {retry} in {wait:g}s")
                sleep(wait)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Model request failed after retries.")


class Provider:
    """Stateful wrapper: current model plus fallbacks over one client."""

    def __init__(
        self,
        *,
        model: str,
        fallbacks: Sequence[str] = (),
        client: Any,
        log: Log = print,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.model = model
        self.fallbacks = list(fallbacks)
        self._client = client
        self._log = log
        self._sleep = sleep

    def respond(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        body: dict[str, Any] = {"messages": list(messages)}
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        models = unique([self.model, *self.fallbacks])
        response, model = create_with_retry(
            self._client, body, models, self._log, self._sleep
        )
        self.model = model
        return response


def create_provider(
    settings: Any | None = None,
    *,
    log: Log = print,
    client: Any | None = None,
    sleep: Sleep = time.sleep,
) -> Provider:
    """Build a Provider from a settings object (default: the config module).

    Settings attributes: API_KEY, BASE_URL, MODEL, FALLBACK_MODELS.
    """
    if settings is None:
        import config as settings  # noqa: PLC0415

    api_key = require_setting(getattr(settings, "API_KEY", None), "API_KEY")
    base_url = require_setting(getattr(settings, "BASE_URL", None), "BASE_URL")
    model = require_setting(getattr(settings, "MODEL", None), "MODEL")
    fallbacks = unique(getattr(settings, "FALLBACK_MODELS", None) or [])

    if client is None:
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    return Provider(
        model=model, fallbacks=fallbacks, client=client, log=log, sleep=sleep
    )
