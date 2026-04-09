import re
import json
import uuid
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _get_field(obj, field: str, default=None):
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _flatten_text_content(content) -> str:
    """Normalize Claude content blocks into plain text."""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    text_parts = []
    for block in content:
        block_type = _get_field(block, "type")
        if block_type == "text":
            text = _get_field(block, "text", "")
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


def extract_search_query(messages: list) -> Optional[str]:
    """Extract search query from Claude Code's web search sub-request."""
    for msg in reversed(messages):
        if _get_field(msg, "role") == "user":
            content = _flatten_text_content(_get_field(msg, "content", ""))
            if not content:
                continue

            match = re.search(
                r"Perform a web search for the query:\s*(.+)$",
                content,
                re.IGNORECASE | re.MULTILINE,
            )
            if match:
                return match.group(1).strip()
    return None


def is_web_search_request(tools: list, system: str) -> bool:
    """Detect if this is Claude Code's web search sub-request."""
    has_web_search_tool = any(
        _get_field(t, "type") == "web_search_20250305"
        or _get_field(t, "name") == "web_search"
        for t in (tools or [])
    )
    is_search_system = "performing a web search" in (system or "").lower()
    return has_web_search_tool and is_search_system


async def execute_search(query: str, max_results: int = 5) -> list:
    """Execute web search using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS

        def _search():
            with DDGS() as ddgs:
                attempts = [
                    ("html", lambda: ddgs._text_html(query, max_results=max_results)),
                    ("lite", lambda: ddgs._text_lite(query, max_results=max_results)),
                    ("bing", lambda: ddgs.text(query, backend="bing", max_results=max_results)),
                ]

                last_error = None
                for backend, run in attempts:
                    try:
                        results = list(run() or [])
                        logger.info(
                            "Search backend %s returned %d results for: %s",
                            backend,
                            len(results),
                            query,
                        )
                        if results:
                            return results
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "Search backend %s failed for %s: %s",
                            backend,
                            query,
                            exc,
                        )

                if last_error:
                    raise last_error
                return []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _search)
        logger.info(f"DuckDuckGo search returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def format_search_results(query: str, results: list) -> str:
    """Format search results as plain text for Claude Code."""
    if not results:
        return f'Web search results for query: "{query}"\n\nNo results found.\n\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.'

    lines = [f'Web search results for query: "{query}"\n']
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}")
        lines.append(f"URL: {url}")
        if body:
            lines.append(f"{body[:300]}")
        lines.append("")

    lines.append(
        "REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks."
    )
    return "\n".join(lines)


async def generate_search_stream(query: str, model: str):
    """
    Generate a Claude-format streaming response for a web search sub-request.
    Executes the search and returns results as server_tool_use + web_search_tool_result blocks.
    """
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:16]}"
    results = await execute_search(query)
    result_text = format_search_results(query, results)

    # Build web search result content
    web_results = []
    for r in results:
        web_results.append(
            {
                "type": "web_search_result",
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "encrypted_content": r.get("body", "")[:500],
                "page_age": None,
            }
        )

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def event(data: dict) -> str:
        return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # message_start
    yield event(
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    )

    # server_tool_use block (Claude Code counts this as a search)
    yield event(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {},
            },
        }
    )
    yield event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"query": query}, ensure_ascii=False),
            },
        }
    )
    yield event({"type": "content_block_stop", "index": 0})

    # web_search_tool_result block (actual search results)
    yield event(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": web_results if web_results else [],
            },
        }
    )
    yield event({"type": "content_block_stop", "index": 1})

    # message_delta
    yield event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 50},
        }
    )

    # message_stop
    yield event({"type": "message_stop"})
