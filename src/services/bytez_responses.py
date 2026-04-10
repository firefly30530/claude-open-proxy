import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional
import logging

import httpx
from fastapi import HTTPException

from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger(__name__)

MIN_BYTEZ_REASONING_TOKENS = 2048


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def model_supports_bytez_responses(model: str) -> bool:
    return (model or "").startswith("anthropic/")


def should_use_bytez_responses_backend(
    request: ClaudeMessagesRequest, mapped_model: str
) -> bool:
    return (
        not request.tools
        and config.bytez_responses_base_url is not None
        and model_supports_bytez_responses(mapped_model)
    )


def _extract_system_text(system) -> Optional[str]:
    if not system:
        return None
    if isinstance(system, str):
        return system

    parts = []
    for block in system:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for block in content:
        block = _dump_model(block)
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif block_type in {"thinking", "redacted_thinking"}:
            continue
        elif block_type == "tool_result":
            result = block.get("content")
            if isinstance(result, str):
                parts.append(result)
            elif result is not None:
                parts.append(json.dumps(result, ensure_ascii=False))
        elif block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            parts.append(
                f"[Tool: {tool_name}] {json.dumps(tool_input, ensure_ascii=False)}"
            )

    return "\n".join(part for part in parts if part).strip()


def build_bytez_responses_request(
    request: ClaudeMessagesRequest, mapped_model: str
) -> Dict[str, Any]:
    max_output_tokens = min(
        max(request.max_tokens, config.min_tokens_limit),
        config.max_tokens_limit,
    )

    input_items = []
    for message in request.messages:
        content = _flatten_content(message.content)
        if not content:
            continue
        input_items.append({"role": message.role, "content": content})

    body: Dict[str, Any] = {
        "model": mapped_model,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
        "stream": request.stream,
    }

    instructions = _extract_system_text(request.system)
    if instructions:
        body["instructions"] = instructions
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata:
        body["metadata"] = request.metadata
    if request.stop_sequences:
        body["stop"] = request.stop_sequences

    request_thinking = request.thinking
    reasoning_enabled = (
        config.bytez_responses_reasoning_enabled
        and request_thinking is not None
        and request_thinking.enabled is not False
    )

    if reasoning_enabled:
        if body["max_output_tokens"] < MIN_BYTEZ_REASONING_TOKENS:
            body["max_output_tokens"] = MIN_BYTEZ_REASONING_TOKENS

        reasoning_payload: Dict[str, Any] = {"summary": "auto"}
        effort = config.bytez_responses_reasoning_effort

        thinking_type = (request_thinking.type or "").lower()
        if thinking_type in {"low", "medium", "high"}:
            effort = thinking_type
        elif thinking_type in {"adaptive", "enabled"}:
            effort = "medium"

        if effort:
            reasoning_payload["effort"] = effort

        budget_tokens = None
        if request_thinking.budget_tokens is not None:
            budget_tokens = request_thinking.budget_tokens
        elif config.bytez_responses_reasoning_budget_tokens is not None:
            budget_tokens = config.bytez_responses_reasoning_budget_tokens
        if budget_tokens is None:
            budget_tokens = MIN_BYTEZ_REASONING_TOKENS
        reasoning_payload["budget_tokens"] = max(
            budget_tokens, MIN_BYTEZ_REASONING_TOKENS
        )

        body["reasoning"] = reasoning_payload

    logger.info(
        "Prepared Bytez Responses request: "
        f"model={mapped_model}, "
        f"stream={request.stream}, "
        f"max_output_tokens={body['max_output_tokens']}, "
        f"thinking_present={request_thinking is not None}, "
        f"thinking_enabled={None if request_thinking is None else request_thinking.enabled}, "
        f"thinking_type={None if request_thinking is None else request_thinking.type}, "
        f"thinking_budget_tokens={None if request_thinking is None else request_thinking.budget_tokens}, "
        f"sending_reasoning={'reasoning' in body}, "
        f"reasoning_effort={body.get('reasoning', {}).get('effort') if 'reasoning' in body else None}, "
        f"reasoning_budget_tokens={body.get('reasoning', {}).get('budget_tokens') if 'reasoning' in body else None}"
    )

    return body


def _extract_content_blocks(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []

    for item in response_json.get("output", []) or []:
        item = _dump_model(item)
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "reasoning":
            summary_parts = []
            for part in item.get("summary", []) or []:
                part = _dump_model(part)
                if isinstance(part, dict) and part.get("type") == "summary_text":
                    text = part.get("text", "")
                    if text:
                        summary_parts.append(text)
            if summary_parts:
                content.append(
                    {"type": "thinking", "thinking": "\n".join(summary_parts)}
                )
            encrypted_content = item.get("encrypted_content")
            if encrypted_content:
                content.append(
                    {"type": "redacted_thinking", "data": encrypted_content}
                )
        elif item_type == "message" and item.get("role") == "assistant":
            for part in item.get("content", []) or []:
                part = _dump_model(part)
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "output_text":
                    text = part.get("text", "")
                    if text:
                        content.append({"type": "text", "text": text})
                elif part_type == "output_refusal":
                    text = part.get("refusal") or part.get("text") or ""
                    if text:
                        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    return content


def _headers() -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {config.openai_api_key}"}
    headers.update(config.get_custom_headers())
    return headers


def _responses_url() -> str:
    return f"{config.bytez_responses_base_url.rstrip('/')}/responses"


def _stop_reason(response_json: Dict[str, Any]) -> str:
    response = _dump_model(response_json) if not isinstance(response_json, dict) else response_json
    status = response.get("status")
    if status == "incomplete":
        return Constants.STOP_MAX_TOKENS
    return Constants.STOP_END_TURN


def _estimate_input_tokens(body: Dict[str, Any]) -> int:
    input_items = body.get("input", []) or []
    input_json = json.dumps(input_items, ensure_ascii=False)
    instructions = body.get("instructions")
    instructions_len = len(instructions) if isinstance(instructions, str) else 0

    # Bytez Responses does not provide prompt token usage until completion.
    # Keep Anthropic-style stream usage stable by using the same up-front estimate
    # in both message_start and message_delta.
    return max(1, round(len(input_json) * 0.9 + instructions_len * 0.5))


async def create_bytez_responses_message(
    request: ClaudeMessagesRequest, mapped_model: str
) -> Dict[str, Any]:
    body = build_bytez_responses_request(request, mapped_model)
    async with httpx.AsyncClient(
        timeout=config.bytez_responses_timeout_s,
        headers=_headers(),
    ) as client:
        response = await client.post(_responses_url(), json=body)
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        response_json = response.json()

    usage = response_json.get("usage", {}) or {}
    return {
        "id": response_json.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": request.model,
        "content": _extract_content_blocks(response_json),
        "stop_reason": _stop_reason(response_json),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        },
    }


async def generate_bytez_responses_stream(
    request: ClaudeMessagesRequest, mapped_model: str
):
    body = build_bytez_responses_request(request, mapped_model)
    estimated_input_tokens = _estimate_input_tokens(body)
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    next_block_index = 0
    text_block_index: Optional[int] = None
    thinking_block_index: Optional[int] = None
    text_block_open = False
    thinking_block_open = False
    completed_payload = None
    thinking_fragments: List[str] = []

    def event(event_name: str, data: Dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def alloc_index() -> int:
        nonlocal next_block_index
        idx = next_block_index
        next_block_index += 1
        return idx

    def start_text_block():
        nonlocal text_block_index, text_block_open
        if text_block_open:
            return None
        text_block_index = alloc_index()
        text_block_open = True
        return event(
            Constants.EVENT_CONTENT_BLOCK_START,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": text_block_index,
                "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
            },
        )

    def stop_text_block():
        nonlocal text_block_open
        if not text_block_open or text_block_index is None:
            return None
        text_block_open = False
        return event(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_block_index},
        )

    def start_thinking_block():
        nonlocal thinking_block_index, thinking_block_open
        if thinking_block_open:
            return None
        thinking_block_index = alloc_index()
        thinking_block_open = True
        return event(
            Constants.EVENT_CONTENT_BLOCK_START,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_START,
                "index": thinking_block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

    def stop_thinking_block():
        nonlocal thinking_block_open
        if not thinking_block_open or thinking_block_index is None:
            return None
        thinking_block_open = False
        return event(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": thinking_block_index},
        )

    def emit_thinking_signature():
        if thinking_block_index is None or not thinking_fragments:
            return None
        signature = hashlib.sha256("".join(thinking_fragments).encode("utf-8")).hexdigest()
        return event(
            Constants.EVENT_CONTENT_BLOCK_DELTA,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                "index": thinking_block_index,
                "delta": {
                    "type": "signature_delta",
                    "signature": signature,
                },
            },
        )

    yield event(
        Constants.EVENT_MESSAGE_START,
        {
            "type": Constants.EVENT_MESSAGE_START,
            "message": {
                "id": message_id,
                "type": "message",
                "role": Constants.ROLE_ASSISTANT,
                "model": request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        },
    )
    yield event(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    async with httpx.AsyncClient(
        timeout=config.bytez_responses_timeout_s,
        headers=_headers(),
    ) as client:
        async with client.stream("POST", _responses_url(), json=body) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise HTTPException(
                    status_code=response.status_code, detail=error_text.decode()
                )

            current_event = None
            data_lines: List[str] = []

            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())
                    continue
                if line != "":
                    continue
                if not data_lines:
                    current_event = None
                    continue

                raw_data = "\n".join(data_lines)
                data_lines = []
                if raw_data == "[DONE]":
                    break

                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    current_event = None
                    continue

                event_type = payload.get("type") or current_event
                current_event = None

                if event_type in {
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                }:
                    delta = payload.get("delta", "")
                    if delta:
                        thinking_fragments.append(delta)
                        maybe = stop_text_block()
                        if maybe:
                            yield maybe
                        maybe = start_thinking_block()
                        if maybe:
                            yield maybe
                        yield event(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": thinking_block_index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": delta,
                                },
                            },
                        )
                elif event_type == "response.output_text.delta":
                    delta = payload.get("delta", "")
                    if delta:
                        maybe = emit_thinking_signature()
                        if maybe:
                            yield maybe
                        maybe = stop_thinking_block()
                        if maybe:
                            yield maybe
                        thinking_fragments.clear()
                        maybe = start_text_block()
                        if maybe:
                            yield maybe
                        yield event(
                            Constants.EVENT_CONTENT_BLOCK_DELTA,
                            {
                                "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                "index": text_block_index,
                                "delta": {
                                    "type": Constants.DELTA_TEXT,
                                    "text": delta,
                                },
                            },
                        )
                elif event_type == "response.completed":
                    completed_payload = payload.get("response", {})
                    break
                elif event_type == "response.failed":
                    raise HTTPException(status_code=502, detail=raw_data)

    maybe = emit_thinking_signature()
    if maybe:
        yield maybe
    maybe = stop_thinking_block()
    if maybe:
        yield maybe
    thinking_fragments.clear()
    maybe = stop_text_block()
    if maybe:
        yield maybe

    usage = (completed_payload or {}).get("usage", {}) if completed_payload else {}
    yield event(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {
                "stop_reason": _stop_reason(completed_payload or {}),
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": estimated_input_tokens,
                "output_tokens": usage.get(
                    "output_tokens", usage.get("completion_tokens", 0)
                ),
            },
        },
    )
    yield event(Constants.EVENT_MESSAGE_STOP, {"type": Constants.EVENT_MESSAGE_STOP})
