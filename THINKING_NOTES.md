# Thinking / Reasoning Notes

This document summarizes the current `thinking` / `reasoning` implementation in this proxy, the observed Claude Code request patterns, the Bytez-specific compatibility behavior, and the main conclusions from debugging.

## Scope

This note is only about:

- Anthropic-style top-level `thinking` on `/v1/messages`
- Translation to Bytez `Responses` `reasoning`
- Why reasoning is difficult to trigger during normal Claude Code main-thread interaction

It does not cover general tool-call conversion except where that directly affects reasoning routing.

## Current Implementation

### 1. Request schema supports Anthropic thinking

`ClaudeMessagesRequest` includes a top-level `thinking` object:

- `enabled`
- `type`
- `budget_tokens`

File:

- `src/models/claude.py`

It also supports Anthropic content blocks:

- `thinking`
- `redacted_thinking`

### 2. Bytez Responses backend exists for no-tool Anthropic requests

The proxy has a dedicated Bytez `Responses` path:

- `src/services/bytez_responses.py`

Current routing rule:

- if `not request.tools`
- and Bytez Responses base URL is configured
- and mapped model starts with `anthropic/`
- then route to Bytez `Responses`

Current entry point:

- `src/api/endpoints.py`

### 3. Anthropic thinking -> Bytez reasoning mapping

When the request is routed to Bytez `Responses` and top-level `thinking` is present and enabled:

- `thinking.type=low` -> `reasoning.effort=low`
- `thinking.type=medium` -> `reasoning.effort=medium`
- `thinking.type=high` -> `reasoning.effort=high`
- `thinking.type=adaptive` -> `reasoning.effort=medium`
- `thinking.type=enabled` -> `reasoning.effort=medium`
- `thinking.budget_tokens` -> `reasoning.budget_tokens`

If `thinking` is absent:

- reasoning is not auto-enabled

If `thinking.enabled=false`:

- reasoning is not sent

### 4. Current minimum token floor for Bytez reasoning

Bytez rejected lower values during testing, so the proxy now clamps reasoning-related token floors to `2048`.

When `thinking` is enabled on the Bytez `Responses` path:

- `max_output_tokens < 2048` -> bumped to `2048`
- `reasoning.budget_tokens < 2048` -> bumped to `2048`
- missing `reasoning.budget_tokens` -> defaulted to `2048`

File:

- `src/services/bytez_responses.py`

### 5. Streaming and non-streaming reasoning extraction

The proxy converts Bytez `Responses` reasoning back into Anthropic-style output:

- non-streaming: emits `thinking` / `redacted_thinking` content blocks
- streaming: emits Anthropic-style `thinking_delta` SSE blocks

This logic is implemented in:

- `src/services/bytez_responses.py`

## Current Config Semantics

Relevant `.env` settings:

- `BYTEZ_RESPONSES_BASE_URL`
- `BYTEZ_RESPONSES_TIMEOUT_S`
- `BYTEZ_RESPONSES_REASONING_ENABLED`
- `BYTEZ_RESPONSES_REASONING_EFFORT`
- `BYTEZ_RESPONSES_REASONING_BUDGET_TOKENS`

Current meaning:

- `BYTEZ_RESPONSES_REASONING_ENABLED`
  - global gate only
  - if false, proxy drops reasoning even if request includes `thinking`
- `BYTEZ_RESPONSES_REASONING_EFFORT`
  - fallback only when request enables `thinking` but does not specify a type
- `BYTEZ_RESPONSES_REASONING_BUDGET_TOKENS`
  - still acts as a config fallback, but the runtime minimum clamp is now `2048`

Important behavioral detail:

- this proxy no longer auto-enables reasoning just because the env flag is true
- a request must actually include top-level `thinking`

## Logging Added

### 1. Bytez Responses request logging

The proxy logs exactly what it is about to send to Bytez `Responses`, including:

- `max_output_tokens`
- `thinking_present`
- `thinking_enabled`
- `thinking_type`
- `thinking_budget_tokens`
- `sending_reasoning`
- `reasoning_effort`
- `reasoning_budget_tokens`

File:

- `src/services/bytez_responses.py`

### 2. Request entry logging

The proxy now logs request-routing-relevant fields at the `/v1/messages` entrypoint:

- `model`
- `mapped_model`
- `stream`
- `has_tools`
- `tool_count`
- `tool_names`
- `thinking_present`
- `thinking_enabled`
- `thinking_type`
- `thinking_budget_tokens`
- `responses_candidate`

File:

- `src/api/endpoints.py`

This is the most important logging added during debugging because it distinguishes:

- requests that contain `thinking`
- requests that contain `tools`
- requests that are eligible for the Bytez `Responses` route

## What We Confirmed by Reproduction

### 1. Reasoning path itself works

Using direct local test requests against the proxy:

- if a request has no tools
- and includes top-level `thinking`
- and reasoning token floors are large enough

then the proxy successfully sends `reasoning` to Bytez and returns Anthropic-style `thinking` blocks in the response.

This was confirmed end-to-end after raising the minimum clamp to `2048`.

### 2. Bytez accepted `2048`, not `1024`

Observed behavior during testing:

- lower values failed
- `1024` still failed
- `1025` still failed
- `2048` succeeded

So the proxy now treats `2048` as the practical working minimum for this integration.

### 3. Native Claude Code `ultrathink` can show up as real top-level thinking

Earlier uncertainty was whether Claude Code was only turning `ultrathink` into reminder text.

Latest entry logging shows a main Sonnet request with:

- `thinking_present=True`
- `thinking_enabled=True`
- `thinking_type=adaptive`

So at least in the current observed request shape, Claude Code can send real top-level `thinking` to this proxy.

## What the Logs Now Show

### Main observed pattern

In current Claude Code interaction, two requests often appear near the same user turn:

1. a Haiku request
2. a Sonnet request

Observed behavior:

- the Haiku request is often:
  - `has_tools=False`
  - `responses_candidate=True`
  - `thinking_present=False`
  - therefore `sending_reasoning=False`
- the Sonnet request is often:
  - `thinking_present=True`
  - but also `has_tools=True`
  - therefore `responses_candidate=False`
  - so it is routed to `chat/completions`, not Bytez `Responses`

This is the key reason reasoning does not trigger during normal main-thread interaction.

### Important correction to earlier debugging assumptions

Earlier hypothesis:

- "reasoning is missing because Claude Code did not send thinking"

This is no longer sufficient.

Latest evidence shows:

- the main Sonnet request can include top-level `thinking`
- but it also includes a full tool manifest

So the more accurate statement is:

- reasoning is blocked mostly by the current routing rule, not by the total absence of top-level thinking

## Current Practical Conclusion

Under the current routing strategy:

- `has_tools=True` -> route to `chat/completions`
- `has_tools=False` -> route to Bytez `Responses`

But current Claude Code main-thread Sonnet requests usually carry tools, even if the user says:

- "do not call tools"
- "just answer directly"

That instruction affects model behavior, but does not remove the tool schema from the request payload.

Therefore:

- normal main-thread Sonnet interaction usually does not get Bytez reasoning
- only no-tool requests can use reasoning
- many no-tool background requests are Haiku helper requests that do not include `thinking`

So, in practice:

- thinking may be present on the important Sonnet request
- but the proxy still cannot use it, because that request also carries tools

## Why "Do Not Use Tools" Does Not Solve It

There is a protocol distinction between:

- tools being available in the request schema
- the model actually deciding to call a tool

Claude Code main requests often still include the tool catalog even when the user asks not to use tools.

That means:

- `has_tools=True`
- even if no tool call is ever executed

Current proxy routing only checks the presence of `request.tools`, not whether the model actually uses them later.

## Current Working Mental Model

The current system behaves like this:

- Haiku helper request
  - usually no tools
  - often no thinking
  - can route to `Responses`
  - usually no reasoning sent

- Sonnet main request
  - often has top-level thinking
  - often also has many tools
  - therefore does not route to `Responses`
  - therefore no Bytez reasoning is used

This is why the current integration can feel like:

- "thinking exists in theory"
- "but almost never triggers during normal Claude Code interaction"

## Files Relevant to Thinking / Reasoning

- `src/models/claude.py`
  - top-level `thinking` request schema
  - `thinking` / `redacted_thinking` content blocks

- `src/api/endpoints.py`
  - request entry logging
  - routing decision between Bytez `Responses` and `chat/completions`

- `src/services/bytez_responses.py`
  - Anthropic `thinking` -> Bytez `reasoning`
  - reasoning token floor clamp
  - non-streaming + streaming reasoning extraction

- `tests/test_bytez_responses.py`
  - tests for routing and reasoning payload construction

## Current Open Problems

### 1. Main-thread reasoning is blocked by tools

This is the primary remaining architectural issue.

### 2. Current routing is too coarse

The proxy currently routes by:

- "tools exist" vs "tools absent"

That is too blunt for Claude Code, because main-thread requests can carry tools even when the user wants a direct answer.

### 3. There is still no policy for "thinking + tools"

Right now the proxy has no special strategy for:

- requests that include real top-level `thinking`
- and also include the full tool schema

## Next-Step Options

### Option A: Thinking-first route

If:

- `thinking_present=True`
- and the request has not yet entered a tool loop

then prefer Bytez `Responses` even if `tools` are present.

Tradeoff:

- this may suppress real tool use on that turn

### Option B: Two-stage planner/executor

1. use Bytez `Responses` for thinking/planning
2. then use `chat/completions` for tool-enabled execution

Tradeoff:

- more complex
- more latency
- more orchestration state

### Option C: Keep current route, accept limited reasoning

This preserves tool behavior, but means:

- normal Claude Code main-thread interaction rarely gets reasoning

## Bottom Line

The proxy now supports Anthropic-style `thinking` correctly on the Bytez `Responses` path, and the Bytez reasoning path itself works when exercised directly.

However, current Claude Code main-thread Sonnet requests usually include both:

- top-level `thinking`
- a full tool manifest

Because the proxy currently treats any tool-enabled request as ineligible for Bytez `Responses`, normal Claude Code interaction still rarely gets real reasoning end-to-end.
