"""vLLM client tool-calling — verified against a stubbed OpenAI response.

No network: we swap the SDK's `chat.completions.create` for a stub, so we can
assert exactly what gets sent (tools) and how the reply is parsed (tool_calls ->
ACTION steps). This is the seam that decides whether a real model can use tools
at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.llm.vllm import VLLMClient
from common.models import LLMMessage, LLMRequest, ReActStepType

_TOOLS = [
    {
        "type": "function",
        "function": {"name": "places_search", "description": "find places", "parameters": {}},
    }
]


def _completion(*, content: str | None, tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        model="geoagent-model",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _client_with(completion: Any) -> tuple[VLLMClient, dict[str, Any]]:
    """Return a client whose `create` records its kwargs and returns `completion`."""
    client = VLLMClient(base_url="http://stub/v1", api_key="k", model="geoagent-model")
    captured: dict[str, Any] = {}

    async def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return completion

    client._client.chat.completions.create = _create  # type: ignore[method-assign]
    return client, captured


def _client_with_reasoning(
    completion: Any, *, effort: str | None = None, max_tokens: int | None = None
) -> tuple[VLLMClient, dict[str, Any]]:
    client = VLLMClient(
        base_url="http://stub/v1",
        api_key="k",
        model="geoagent-model",
        reasoning_effort=effort,
        reasoning_max_tokens=max_tokens,
    )
    captured: dict[str, Any] = {}

    async def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return completion

    client._client.chat.completions.create = _create  # type: ignore[method-assign]
    return client, captured


class _AsyncChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk


class _FailingChunks:
    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        raise RuntimeError("model is temporarily rate-limited upstream")
        yield  # pragma: no cover


class _NeverEndingChunks:
    """Keep yielding SSE activity so a per-read timeout would never fire."""

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        while True:
            yield _stream_chunk()
            await asyncio.sleep(0)


def _stream_chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: Any = None,
    finish_reason: str | None = None,
) -> Any:
    choices = []
    if content is not None or tool_calls is not None or finish_reason is not None:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(model="geoagent-model", choices=choices, usage=usage)


async def test_tools_are_forwarded_to_the_server():
    # The whole point: without this the model is never told the tools exist.
    client, captured = _client_with(_completion(content="hi"))

    await client.generate(LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS))

    assert captured["tools"] == _TOOLS
    assert captured["tool_choice"] == "auto"


async def test_no_tools_means_no_tool_choice():
    # `tool_choice` with an empty tool list is rejected by the API.
    client, captured = _client_with(_completion(content="hi"))

    await client.generate(LLMRequest(messages=[LLMMessage(role="user", content="q")]))

    assert "tools" not in captured
    assert "tool_choice" not in captured


async def test_sampling_parameters_are_forwarded_to_the_server():
    client, captured = _client_with(_completion(content="hi"))

    await client.generate(
        LLMRequest(
            messages=[LLMMessage(role="user", content="q")],
            temperature=0.2,
            top_p=0.9,
        )
    )

    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9


async def test_reasoning_effort_is_forwarded_to_non_streaming_request():
    client, captured = _client_with_reasoning(_completion(content="hi"), effort="low")

    response = await client.generate(LLMRequest(messages=[LLMMessage(role="user", content="q")]))

    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}
    assert response.finish_reason == "stop"


async def test_reasoning_max_tokens_is_forwarded_to_non_streaming_request():
    client, captured = _client_with_reasoning(_completion(content="hi"), max_tokens=800)

    await client.generate(LLMRequest(messages=[LLMMessage(role="user", content="q")]))

    assert captured["extra_body"] == {"reasoning": {"max_tokens": 800}}


async def test_tool_calls_become_action_steps():
    completion = _completion(
        content="Let me look that up.",
        tool_calls=[_tool_call("call_1", "places_search", '{"mode":"area","query":"кафе"}')],
    )
    client, _ = _client_with(completion)

    response = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS)
    )

    steps = response.trace.steps
    assert [s.type for s in steps] == [ReActStepType.THOUGHT, ReActStepType.ACTION]
    action = steps[1]
    assert action.tool_name == "places_search"
    assert action.tool_input == {"mode": "area", "query": "кафе"}
    assert action.tool_call_id == "call_1"
    # A tool turn is not a final answer — the loop must continue.
    assert response.trace.final_answer is None


async def test_parallel_tool_calls_all_become_actions():
    completion = _completion(
        content=None,
        tool_calls=[
            _tool_call("c1", "places_search", "{}"),
            _tool_call("c2", "web_search", '{"query":"x"}'),
        ],
    )
    client, _ = _client_with(completion)

    response = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS)
    )

    actions = [s for s in response.trace.steps if s.type is ReActStepType.ACTION]
    assert [a.tool_name for a in actions] == ["places_search", "web_search"]
    assert [a.tool_call_id for a in actions] == ["c1", "c2"]


async def test_malformed_arguments_degrade_to_empty_dict():
    # Broken JSON is data: the tool's own validation reports INVALID_INPUT and the
    # model sees a normal error observation instead of the request crashing.
    completion = _completion(
        content=None, tool_calls=[_tool_call("c1", "places_search", "{not json")]
    )
    client, _ = _client_with(completion)

    response = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS)
    )

    action = next(s for s in response.trace.steps if s.type is ReActStepType.ACTION)
    assert action.tool_input == {}


async def test_plain_completion_is_a_final_answer():
    client, _ = _client_with(_completion(content="Here is your plan."))

    response = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS)
    )

    assert response.trace.final_answer == "Here is your plan."
    assert [s.type for s in response.trace.steps] == [ReActStepType.FINAL_ANSWER]
    assert response.usage.total_tokens == 10


async def test_empty_completion_is_not_classified_as_a_final_answer():
    client, _ = _client_with(_completion(content=None))

    response = await client.generate(
        LLMRequest(messages=[LLMMessage(role="user", content="q")], tools=_TOOLS)
    )

    assert response.content == ""
    assert response.trace.steps == []
    assert response.trace.final_answer is None


async def test_native_stream_forwards_text_and_rebuilds_response() -> None:
    usage = SimpleNamespace(
        prompt_tokens=7,
        completion_tokens=2,
        total_tokens=9,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
    )
    chunks = _AsyncChunks(
        [
            _stream_chunk(content="Here "),
            _stream_chunk(content="you go.", finish_reason="stop"),
            _stream_chunk(usage=usage),
        ]
    )
    client, captured = _client_with(chunks)
    emitted: list[str] = []

    response = await client.generate_stream(
        LLMRequest(messages=[LLMMessage(role="user", content="q")]), emitted.append
    )

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert emitted == ["Here ", "you go."]
    assert response.content == "Here you go."
    assert response.trace.final_answer == "Here you go."
    assert response.usage.total_tokens == 9
    assert response.usage.reasoning_tokens == 1
    assert response.finish_reason == "stop"


async def test_stream_timeout_is_a_deadline_for_the_complete_stream() -> None:
    client = VLLMClient(
        base_url="http://stub/v1",
        api_key="k",
        model="geoagent-model",
        timeout=1,
    )

    async def _create(**_kwargs: Any) -> Any:
        return _NeverEndingChunks()

    client._client.chat.completions.create = _create  # type: ignore[method-assign]

    with pytest.raises(TimeoutError):
        await client.generate_stream(
            LLMRequest(messages=[LLMMessage(role="user", content="q")]),
            lambda _chunk: None,
        )


async def test_stream_retries_one_rate_limit_before_any_output(monkeypatch) -> None:
    client = VLLMClient(base_url="http://stub/v1", api_key="k", model="geoagent-model")
    responses = iter(
        [
            _FailingChunks(),
            _AsyncChunks([_stream_chunk(content="done", finish_reason="stop")]),
        ]
    )
    calls = 0

    async def _create(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return next(responses)

    async def _no_sleep(_delay: float) -> None:
        return None

    client._client.chat.completions.create = _create  # type: ignore[method-assign]
    monkeypatch.setattr("backend.app.llm.vllm.asyncio.sleep", _no_sleep)
    emitted: list[str] = []

    response = await client.generate_stream(
        LLMRequest(messages=[LLMMessage(role="user", content="q")]), emitted.append
    )

    assert calls == 2
    assert emitted == ["done"]
    assert response.content == "done"


async def test_reasoning_effort_is_forwarded_to_streaming_request() -> None:
    chunks = _AsyncChunks([_stream_chunk(content="done", finish_reason="stop")])
    client, captured = _client_with_reasoning(chunks, effort="low")

    await client.generate_stream(
        LLMRequest(messages=[LLMMessage(role="user", content="q")]),
        lambda _chunk: None,
    )

    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}


async def test_reasoning_max_tokens_is_forwarded_to_streaming_request() -> None:
    chunks = _AsyncChunks([_stream_chunk(content="done", finish_reason="stop")])
    client, captured = _client_with_reasoning(chunks, max_tokens=800)

    await client.generate_stream(
        LLMRequest(messages=[LLMMessage(role="user", content="q")]),
        lambda _chunk: None,
    )

    assert captured["extra_body"] == {"reasoning": {"max_tokens": 800}}


async def test_tool_turns_are_sent_in_the_openai_wire_format():
    client, captured = _client_with(_completion(content="ok"))

    await client.generate(
        LLMRequest(
            messages=[
                LLMMessage(role="user", content="q"),
                LLMMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "places_search", "arguments": "{}"},
                        }
                    ],
                ),
                LLMMessage(role="tool", content='{"places":[]}', tool_call_id="c1"),
            ],
            tools=_TOOLS,
        )
    )

    sent = captured["messages"]
    assert sent[1]["tool_calls"][0]["id"] == "c1"
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "c1"
    # Plain messages must not grow empty tool fields.
    assert "tool_calls" not in sent[0]
