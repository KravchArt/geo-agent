"""Real LLM client over an OpenAI-compatible vLLM server.

Not exercised in CI (LLM_MODE=mock there). It exists so that switching to a real
model is a config change only. It returns the SAME :class:`LLMResponse` shape as
the mock client.

See README ("vLLM — reference") for how to launch a compatible server.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from backend.app.llm.base import LLMClient
from common.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReActStep,
    ReActStepType,
    ReActTrace,
)


def _to_openai_messages(messages: Sequence[LLMMessage]) -> list[ChatCompletionMessageParam]:
    """Our message contract -> the OpenAI wire format, tool turns included."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        payload.append(item)
    return cast(list[ChatCompletionMessageParam], payload)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string*; malformed JSON is data, not a crash.

    An empty dict still reaches the tool, whose own validation rejects it as
    ``INVALID_INPUT`` — the model then sees a normal error observation.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _usage_from_openai(raw_usage: Any | None) -> LLMUsage:
    """Normalize usage, retaining OpenRouter's separate reasoning count."""

    if raw_usage is None:
        return LLMUsage()
    details = getattr(raw_usage, "completion_tokens_details", None)
    return LLMUsage(
        prompt_tokens=raw_usage.prompt_tokens,
        completion_tokens=raw_usage.completion_tokens,
        total_tokens=raw_usage.total_tokens,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
    )


def _is_transient_rate_limit_error(exc: Exception) -> bool:
    """Recognize provider rate limits, including OpenRouter's generic APIError."""

    if getattr(exc, "status_code", None) == 429:
        return True
    message = str(exc).casefold()
    return "temporarily rate-limited" in message or "rate limit" in message


def _trace_from_message(message: Any) -> ReActTrace:
    """Build a ReActTrace from one assistant message.

    ``tool_calls`` present -> ACTION step per call (the loop executes them);
    non-empty text without tool calls is the final answer. An empty completion
    stays unclassified so the orchestrator can retry or return an explicit fallback.
    """
    content = message.content or ""
    tool_calls = getattr(message, "tool_calls", None) or []

    if not tool_calls:
        if not content.strip():
            return ReActTrace()
        return ReActTrace(
            steps=[ReActStep(type=ReActStepType.FINAL_ANSWER, content=content)],
            final_answer=content,
        )

    steps: list[ReActStep] = []
    # Models often narrate before calling; keep it as the thought.
    if content.strip():
        steps.append(ReActStep(type=ReActStepType.THOUGHT, content=content))
    for call in tool_calls:
        steps.append(
            ReActStep(
                type=ReActStepType.ACTION,
                content=f"Call {call.function.name}.",
                tool_name=call.function.name,
                tool_input=_parse_arguments(call.function.arguments),
                tool_call_id=call.id,
            )
        )
    return ReActTrace(steps=steps)


class VLLMClient(LLMClient):
    """OpenAI-compatible client pointed at a vLLM server."""

    mode = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: int = 60,
        proxy: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_max_tokens: int | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_max_tokens is not None:
            raise ValueError("reasoning effort and max tokens are mutually exclusive")
        # vLLM ignores the key unless started with --api-key; "EMPTY" is the
        # conventional placeholder the OpenAI SDK requires to be non-empty.
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=float(timeout),
            # The SDK would otherwise build an httpx client that reads
            # HTTP_PROXY/ALL_PROXY from the environment. The model server is
            # addressed explicitly by base_url; a shell variable must not
            # silently reroute it (or break startup on a socks:// ALL_PROXY).
            http_client=httpx.AsyncClient(timeout=float(timeout), proxy=proxy, trust_env=False),
        )
        self.model = model
        # HTTPX guards individual socket operations. Keep the same setting as a
        # hard wall-clock deadline for the complete LLM call as well.
        self._timeout = float(timeout)
        self._reasoning_effort = reasoning_effort
        self._reasoning_max_tokens = reasoning_max_tokens

    def _request_options(self, request: LLMRequest) -> dict[str, Any]:
        """Return optional tool and reasoning controls for one provider call."""

        extra: dict[str, Any] = {}
        if request.tools:
            extra["tools"] = request.tools
            extra["tool_choice"] = "auto"
        if self._reasoning_effort is not None:
            extra["extra_body"] = {"reasoning": {"effort": self._reasoning_effort}}
        elif self._reasoning_max_tokens is not None:
            extra["extra_body"] = {"reasoning": {"max_tokens": self._reasoning_max_tokens}}
        return extra

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Only advertise tools when there are any: `tool_choice` on an empty list
        # is rejected by the API.
        extra = self._request_options(request)
        async with asyncio.timeout(self._timeout):
            completion = await self._client.chat.completions.create(
                model=request.model or self.model,
                messages=_to_openai_messages(request.messages),
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                **extra,
            )
        message = completion.choices[0].message
        usage = _usage_from_openai(completion.usage)
        return LLMResponse(
            model=completion.model,
            mode="vllm",
            content=message.content or "",
            trace=_trace_from_message(message),
            usage=usage,
            finish_reason=completion.choices[0].finish_reason,
        )

    async def generate_stream(
        self,
        request: LLMRequest,
        on_text_chunk: Callable[[str], None],
    ) -> LLMResponse:
        """Consume the entire stream within the configured wall-clock deadline."""
        async with asyncio.timeout(self._timeout):
            return await self._generate_stream(request, on_text_chunk)

    async def _generate_stream(
        self,
        request: LLMRequest,
        on_text_chunk: Callable[[str], None],
    ) -> LLMResponse:
        """Consume OpenAI chat deltas and rebuild the ordinary response shape."""
        extra = self._request_options(request)
        content_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        usage = LLMUsage()
        response_model = request.model or self.model
        finish_reason: str | None = None

        for attempt in range(2):
            try:
                stream = await self._client.chat.completions.create(
                    model=request.model or self.model,
                    messages=_to_openai_messages(request.messages),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    **extra,
                )
                async for chunk in stream:
                    response_model = chunk.model or response_model
                    raw_usage = chunk.usage
                    if raw_usage is not None:
                        usage = _usage_from_openai(raw_usage)
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta.content:
                        content_parts.append(delta.content)
                        on_text_chunk(delta.content)
                    for call in delta.tool_calls or []:
                        part = tool_parts.setdefault(
                            call.index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if call.id:
                            part["id"] += call.id
                        function = call.function
                        if function is not None:
                            if function.name:
                                part["name"] += function.name
                            if function.arguments:
                                part["arguments"] += function.arguments
                break
            except Exception as exc:
                safe_to_retry = not content_parts and not tool_parts
                if attempt == 0 and safe_to_retry and _is_transient_rate_limit_error(exc):
                    await asyncio.sleep(0.5)
                    continue
                raise

        content = "".join(content_parts)
        tool_calls = [
            SimpleNamespace(
                id=part["id"] or f"call_{index}",
                function=SimpleNamespace(
                    name=part["name"],
                    arguments=part["arguments"],
                ),
            )
            for index, part in sorted(tool_parts.items())
        ]
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return LLMResponse(
            model=response_model,
            mode="vllm",
            content=content,
            trace=_trace_from_message(message),
            usage=usage,
            finish_reason=finish_reason,
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True
