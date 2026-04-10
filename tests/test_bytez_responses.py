from src.services.bytez_responses import (
    build_bytez_responses_request,
    should_use_bytez_responses_backend,
)
from src.models.claude import ClaudeMessagesRequest


def make_request(**overrides):
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ClaudeMessagesRequest.model_validate(payload)


def test_routes_no_tool_requests_to_bytez_responses(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_base_url",
        "https://api.bytez.com/models/v2/openai/v1",
    )

    request = make_request()

    assert should_use_bytez_responses_backend(
        request, "anthropic/claude-sonnet-4-6"
    )


def test_keeps_tool_requests_on_chat_completions(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_base_url",
        "https://api.bytez.com/models/v2/openai/v1",
    )

    request = make_request(
        tools=[
            {
                "name": "calculator",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    )

    assert not should_use_bytez_responses_backend(
        request, "anthropic/claude-sonnet-4-6"
    )


def test_no_thinking_does_not_auto_enable_reasoning(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", True
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_effort", "high"
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_budget_tokens",
        2048,
    )

    body = build_bytez_responses_request(
        make_request(), "anthropic/claude-sonnet-4-6"
    )

    assert "reasoning" not in body


def test_request_thinking_can_disable_reasoning(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", True
    )

    body = build_bytez_responses_request(
        make_request(thinking={"enabled": False}),
        "anthropic/claude-sonnet-4-6",
    )

    assert "reasoning" not in body


def test_global_switch_can_disable_requested_reasoning(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", False
    )

    body = build_bytez_responses_request(
        make_request(thinking={"enabled": True, "type": "high"}),
        "anthropic/claude-sonnet-4-6",
    )

    assert "reasoning" not in body


def test_request_thinking_type_overrides_effort(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", True
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_effort", "low"
    )

    body = build_bytez_responses_request(
        make_request(thinking={"enabled": True, "type": "high", "budget_tokens": 512}),
        "anthropic/claude-sonnet-4-6",
    )

    assert body["reasoning"] == {
        "summary": "auto",
        "effort": "high",
        "budget_tokens": 2048,
    }
    assert body["max_output_tokens"] == 2048


def test_requested_reasoning_without_type_uses_global_fallback_effort(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", True
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_effort", "medium"
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_budget_tokens",
        2048,
    )

    body = build_bytez_responses_request(
        make_request(thinking={"enabled": True}),
        "anthropic/claude-sonnet-4-6",
    )

    assert body["reasoning"] == {
        "summary": "auto",
        "effort": "medium",
        "budget_tokens": 2048,
    }


def test_requested_reasoning_without_budget_uses_minimum_budget(monkeypatch):
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_enabled", True
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_effort", "medium"
    )
    monkeypatch.setattr(
        "src.services.bytez_responses.config.bytez_responses_reasoning_budget_tokens",
        None,
    )

    body = build_bytez_responses_request(
        make_request(thinking={"enabled": True}),
        "anthropic/claude-sonnet-4-6",
    )

    assert body["reasoning"] == {
        "summary": "auto",
        "effort": "medium",
        "budget_tokens": 2048,
    }
    assert body["max_output_tokens"] == 2048
