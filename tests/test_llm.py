"""LLM client: retry, fallback switching, and config validation (no network)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx  # openai>=3 ships its own httpx fork
import pytest
from openai import APIConnectionError, APIStatusError

from agent.llm import (
    ConfigError,
    Provider,
    create_provider,
    create_with_retry,
    is_retryable,
    unique,
)


def status_error(status: int, message: str = "busy") -> APIStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(message, response=response, body=None)


class FakeClient:
    """Scripted chat.completions.create: pops one outcome per call."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self))

    def __call__(self, **request: Any) -> Any:
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Recorder:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.sleeps: list[float] = []

    def log(self, text: str) -> None:
        self.logs.append(text)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class TestHelpers:
    @pytest.mark.parametrize("status", [429, 503])
    def test_retryable_statuses(self, status: int) -> None:
        assert is_retryable(status_error(status))

    @pytest.mark.parametrize("status", [400, 401, 404, 500])
    def test_non_retryable_statuses(self, status: int) -> None:
        assert not is_retryable(status_error(status))

    def test_marker_in_message(self) -> None:
        assert is_retryable(RuntimeError("model UNAVAILABLE right now"))
        assert is_retryable(RuntimeError("high demand, try later"))
        assert not is_retryable(RuntimeError("bad request"))

    def test_connection_error_not_retryable(self) -> None:
        request = httpx.Request("POST", "https://example.test")
        assert not is_retryable(APIConnectionError(request=request))

    def test_unique(self) -> None:
        assert unique(["a", "", None, "b", "a", 3]) == ["a", "b"]


class TestCreateWithRetry:
    def test_success_first_try(self) -> None:
        client = FakeClient(["ok"])
        rec = Recorder()
        result = create_with_retry(client, {"x": 1}, ["m1"], rec.log, rec.sleep)
        assert result == ("ok", "m1")
        assert client.calls == [{"x": 1, "model": "m1"}]
        assert rec.logs == [] and rec.sleeps == []

    def test_429_retries_with_linear_backoff(self) -> None:
        client = FakeClient([status_error(429), status_error(429), "ok"])
        rec = Recorder()
        result = create_with_retry(client, {}, ["m1"], rec.log, rec.sleep)
        assert result == ("ok", "m1")
        assert rec.sleeps == [2.0, 4.0]
        assert rec.logs == [
            "model busy (429); retry 1/2 in 2s",
            "model busy (429); retry 2/2 in 4s",
        ]

    def test_429_exhausted_without_fallback_raises(self) -> None:
        errors = [status_error(429) for _ in range(3)]
        client = FakeClient(errors)
        rec = Recorder()
        with pytest.raises(APIStatusError):
            create_with_retry(client, {}, ["m1"], rec.log, rec.sleep)
        assert len(client.calls) == 3

    def test_429_exhausted_switches_to_fallback(self) -> None:
        errors = [status_error(429) for _ in range(3)]
        client = FakeClient([*errors, "ok"])
        rec = Recorder()
        result = create_with_retry(client, {}, ["m1", "m2"], rec.log, rec.sleep)
        assert result == ("ok", "m2")
        assert [c["model"] for c in client.calls] == ["m1", "m1", "m1", "m2"]
        assert rec.logs[-1] == "model busy (429); switching m1 -> m2"

    def test_503_switches_immediately(self) -> None:
        client = FakeClient([status_error(503), "ok"])
        rec = Recorder()
        result = create_with_retry(client, {}, ["m1", "m2"], rec.log, rec.sleep)
        assert result == ("ok", "m2")
        assert rec.sleeps == []
        assert rec.logs == ["model busy (503); switching m1 -> m2"]

    def test_503_without_fallback_retries(self) -> None:
        client = FakeClient([status_error(503), "ok"])
        rec = Recorder()
        result = create_with_retry(client, {}, ["m1"], rec.log, rec.sleep)
        assert result == ("ok", "m1")
        assert rec.sleeps == [2.0]

    def test_marker_error_uses_503_in_log(self) -> None:
        client = FakeClient([RuntimeError("UNAVAILABLE"), "ok"])
        rec = Recorder()
        create_with_retry(client, {}, ["m1", "m2"], rec.log, rec.sleep)
        assert rec.logs == ["model busy (503); switching m1 -> m2"]

    def test_no_models_raises(self) -> None:
        with pytest.raises(RuntimeError, match="failed after retries"):
            create_with_retry(FakeClient([]), {}, [], lambda _: None)

    def test_non_retryable_raises_immediately(self) -> None:
        client = FakeClient([status_error(401, "bad key"), "ok"])
        rec = Recorder()
        with pytest.raises(APIStatusError, match="bad key"):
            create_with_retry(client, {}, ["m1", "m2"], rec.log, rec.sleep)
        assert len(client.calls) == 1

    def test_all_models_fail_raises_last_error(self) -> None:
        # m1: 503 -> switch; m2 (no fallback left): 3 attempts, then raise
        m2_errors = [
            status_error(503),
            status_error(503),
            status_error(503, "last"),
        ]
        client = FakeClient([status_error(503), *m2_errors])
        rec = Recorder()
        with pytest.raises(APIStatusError, match="last"):
            create_with_retry(client, {}, ["m1", "m2"], rec.log, rec.sleep)
        assert [c["model"] for c in client.calls] == ["m1", "m2", "m2", "m2"]
        assert rec.sleeps == [2.0, 4.0]


class TestProvider:
    def test_respond_builds_body_and_tracks_model(self) -> None:
        client = FakeClient([status_error(503), "ok"])
        rec = Recorder()
        provider = Provider(
            model="m1",
            fallbacks=["m2", "m1"],
            client=client,
            log=rec.log,
            sleep=rec.sleep,
        )
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "t"}}]
        assert provider.respond(messages, tools) == "ok"
        assert provider.model == "m2"
        assert client.calls[-1] == {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "model": "m2",
        }

    def test_respond_without_tools_omits_tool_fields(self) -> None:
        client = FakeClient(["ok"])
        provider = Provider(model="m1", client=client)
        provider.respond([{"role": "user", "content": "hi"}], tools=[])
        assert "tools" not in client.calls[0]
        assert "tool_choice" not in client.calls[0]

    def test_sticks_to_switched_model(self) -> None:
        client = FakeClient([status_error(503), "ok", "ok"])
        provider = Provider(
            model="m1", fallbacks=["m2"], client=client, log=lambda _: None
        )
        provider.respond([])
        provider.respond([])
        assert [c["model"] for c in client.calls] == ["m1", "m2", "m2"]


class TestCreateProvider:
    def settings(self, **overrides: Any) -> SimpleNamespace:
        base = {
            "API_KEY": "key",
            "BASE_URL": "https://example.test/v1",
            "MODEL": "m1",
            "FALLBACK_MODELS": ["m2", "", "m1"],
        }
        return SimpleNamespace(**{**base, **overrides})

    def test_reads_settings(self) -> None:
        client = FakeClient([])
        provider = create_provider(self.settings(), client=client)
        assert provider.model == "m1"
        assert provider.fallbacks == ["m2", "m1"]  # blanks dropped
        assert provider._client is client

    @pytest.mark.parametrize("name", ["API_KEY", "BASE_URL", "MODEL"])
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_missing_setting(self, name: str, bad: object) -> None:
        with pytest.raises(ConfigError, match=f"{name} is not set"):
            create_provider(self.settings(**{name: bad}), client=FakeClient([]))

    def test_no_fallbacks_attribute(self) -> None:
        settings = self.settings()
        del settings.FALLBACK_MODELS
        provider = create_provider(settings, client=FakeClient([]))
        assert provider.fallbacks == []

    def test_default_settings_is_config_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "config", self.settings(MODEL="cfg"))
        provider = create_provider(client=FakeClient([]))
        assert provider.model == "cfg"

    def test_builds_real_client_with_retries_disabled(self) -> None:
        provider = create_provider(self.settings())
        client = provider._client
        assert client.max_retries == 0
        assert str(client.base_url).startswith("https://example.test/v1")
