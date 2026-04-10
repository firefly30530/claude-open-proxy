from src.conversion.request_converter import convert_claude_to_openai
from src.models.claude import ClaudeMessagesRequest


class StubModelManager:
    @staticmethod
    def map_claude_model_to_openai(_model):
        return "anthropic/claude-sonnet-4-6"


def make_request(**overrides):
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ClaudeMessagesRequest.model_validate(payload)


def test_no_thinking_does_not_add_chat_completions_reasoning(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_effort",
        "high",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_budget_tokens",
        2048,
    )

    body = convert_claude_to_openai(
        make_request(),
        StubModelManager(),
    )

    assert "extra_body" not in body


def test_request_thinking_can_disable_chat_completions_reasoning(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )

    body = convert_claude_to_openai(
        make_request(thinking={"enabled": False}),
        StubModelManager(),
    )

    assert "extra_body" not in body


def test_non_bytez_backend_does_not_forward_thinking(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.openai.com/v1",
    )

    body = convert_claude_to_openai(
        make_request(thinking={"enabled": True, "type": "high"}),
        StubModelManager(),
    )

    assert "extra_body" not in body


def test_request_thinking_type_overrides_effort_on_chat_completions(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_effort",
        "low",
    )

    body = convert_claude_to_openai(
        make_request(thinking={"enabled": True, "type": "high", "budget_tokens": 512}),
        StubModelManager(),
    )

    assert body["extra_body"] == {
        "thinking": {
            "type": "high",
            "budget_tokens": 2048,
        },
        "reasoning_effort": "high",
    }
    assert body["max_tokens"] == 2048


def test_requested_reasoning_without_type_uses_global_fallback_effort(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_effort",
        "medium",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_budget_tokens",
        2048,
    )

    body = convert_claude_to_openai(
        make_request(thinking={"enabled": True}),
        StubModelManager(),
    )

    assert body["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2048,
        },
        "reasoning_effort": "medium",
    }
    assert body["max_tokens"] == 2048


def test_thinking_is_forwarded_even_when_tools_are_present(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_budget_tokens",
        None,
    )

    body = convert_claude_to_openai(
        make_request(
            thinking={"enabled": True},
            tools=[
                {
                    "name": "calculator",
                    "description": "Simple math tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "number"}},
                    },
                }
            ],
        ),
        StubModelManager(),
    )

    assert body["tools"][0]["function"]["name"] == "calculator"
    assert body["extra_body"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_adaptive_thinking_degrades_to_enabled(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.request_converter.config.openai_base_url",
        "https://api.bytez.com/v1",
    )
    monkeypatch.setattr(
        "src.conversion.request_converter.config.bytez_responses_reasoning_effort",
        "high",
    )

    body = convert_claude_to_openai(
        make_request(
            max_tokens=256,
            thinking={"enabled": True, "type": "adaptive", "budget_tokens": 4096},
        ),
        StubModelManager(),
    )

    assert body["max_tokens"] == 2048
    assert body["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4096,
        },
        "reasoning_effort": "medium",
    }
