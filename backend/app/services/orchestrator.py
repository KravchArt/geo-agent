"""ReAct orchestrator — the LLM ↔ tools loop for one request.

The gates decide *whether* to run; this decides *how*. It drives the loop the
pipeline was missing: ask the model for the next step, and if that step is a tool
call, execute it (:class:`ToolExecutor`), feed the result back as an observation,
and ask again — until the model produces a final answer or the step budget runs
out.

The model never touches real coordinates or URLs: tools return refs
(``plc_``/``src_``), the model passes refs to the next tool, and adapters expand
them on the way out (see :mod:`tools.refs`).

Backends are interchangeable: with no tools advertised (the lean default) the mock
answers in one turn; with tools configured, the same loop drives either the mock
or a real vLLM model. Nothing here is provider-specific.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from pydantic import ValidationError

from backend.app.llm.base import LLMClient
from backend.app.prompts.orchestrator import build_orchestrator_prompt
from backend.app.services.final_answer_stream import InternalRefStreamFilter
from common.models import (
    ConversationTurn,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReActStep,
    ReActStepType,
    ReActTrace,
)
from tools.base import (
    DEFAULT_MAX_TOOL_ERROR_CHARS,
    ToolErrorCode,
    ToolMetrics,
    ToolResult,
    ToolSpec,
    render_tool_error_observation,
    render_tool_observation,
)
from tools.executor import ToolExecutor
from tools.geo.places_search.category_normalization import (
    category_for_exact_query,
    russian_category_query,
)
from tools.geo.places_search.schemas import (
    OrganisationResolution,
    Place,
    PlacesSearchOutput,
)
from tools.geo.text_place_query import uses_cyrillic

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_REGENERATIONS = 1


def _restore_russian_generic_places_query(
    arguments: dict[str, Any],
    user_message: str,
) -> dict[str, Any]:
    """Undo an English enum-label query emitted for a Russian category request.

    This intentionally handles only an exact generic category label. Named-place
    searches and richer free-text queries remain entirely model-controlled.
    """

    query = arguments.get("query")
    category = arguments.get("category")
    if (
        not isinstance(query, str)
        or not isinstance(category, str)
        or not uses_cyrillic(user_message)
        or uses_cyrillic(query)
        or category_for_exact_query(query) != category
    ):
        return arguments

    corrected = dict(arguments)
    corrected["query"] = russian_category_query(category)
    return corrected


class EchoStore(Protocol):
    """Echo-grounding sink: tool_hash -> the payload the model was shown."""

    async def set_echo(self, tool_hash: str, value: str, ttl: int | None = None) -> None: ...


#: Operational progress only: callers can expose this in a UI without leaking
#: model chain-of-thought. ``kind`` is one of model_started, tool_started,
#: tool_finished, answer_validation, answer_reset, or answer_ready.
ProgressCallback = Callable[[str, dict[str, Any]], None]
AnswerChunkCallback = Callable[[str], None]


#: Keep unexpectedly verbose provider errors from consuming the model context.
_MAX_TOOL_ERROR_CHARS = DEFAULT_MAX_TOOL_ERROR_CHARS

#: Earlier turns replayed to the model. Without them a follow-up ("а что рядом?")
#: arrives with nothing to resolve "рядом" against, and refs from the previous
#: answer cannot be reused — the model geocodes the same place again.
_HISTORY_TURNS = 8

_WEB_RESULTS_FOLLOW_UP = {
    "message": "The web search completed successfully and returned relevant results.",
    "agent_guidance": {
        "retry_policy": (
            "Do not repeat or reformulate the same web search unless a separate user "
            "sub-question remains unresolved."
        ),
        "next_action": (
            "Use the returned evidence or named candidates immediately. For this same part of "
            "the request, places_search may only use mode=resolve on those candidates; never use "
            "mode=area or mode=near to discover, replace, expand, or pad them because those modes "
            "cannot preserve the web-only criterion. Before including a resolved candidate, "
            "verify that its resolved place type or category matches the entity type requested. "
            "Cite every web-supported claim with its exact returned [[src_...]] ref."
        ),
    },
}

_WEB_SEARCH_STOP_WORDS = frozenset(
    {
        "a",
        "and",
        "find",
        "for",
        "in",
        "list",
        "of",
        "search",
        "show",
        "the",
        "to",
        "в",
        "во",
        "для",
        "и",
        "из",
        "на",
        "найди",
        "найти",
        "по",
        "покажи",
        "с",
        "список",
        "со",
    }
)
_WEB_SEARCH_TOKEN_ALIASES = {
    "руб": "рубль",
    "рубля": "рубль",
    "рублей": "рубль",
    "счет": "чек",
    "счета": "чек",
    "счетом": "чек",
}
_RU_QUERY_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ых",
    "их",
    "ий",
    "ый",
    "ая",
    "яя",
    "ое",
    "ее",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "им",
    "ым",
    "ы",
    "и",
    "а",
    "я",
    "е",
    "у",
    "о",
)


def _light_web_search_stem(token: str) -> str:
    """Normalize common Russian inflections without adding an NLP dependency."""

    if not re.fullmatch(r"[а-я]+", token):
        return token[:-1] if token.endswith("s") and len(token) > 4 else token
    for suffix in _RU_QUERY_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _web_search_query_parts(query: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return normalized semantic tokens and strict numeric constraints."""

    normalized = unicodedata.normalize("NFKC", query).casefold().replace("ё", "е")
    normalized = normalized.replace("₽", " рубль ")
    raw_tokens = [token.strip("_") for token in re.findall(r"\w+", normalized)]
    numbers = frozenset(token for token in raw_tokens if token.isdecimal())
    words = frozenset(
        _light_web_search_stem(_WEB_SEARCH_TOKEN_ALIASES.get(token, token))
        for token in raw_tokens
        if token and not token.isdecimal() and token not in _WEB_SEARCH_STOP_WORDS
    )
    return words, numbers


def _web_search_filters(arguments: dict[str, Any]) -> tuple[Any, ...]:
    """Return provider parameters that materially change a web search."""

    return (
        arguments.get("topic") or "general",
        arguments.get("time_range"),
        tuple(sorted(str(value).casefold() for value in (arguments.get("include_domains") or ()))),
        tuple(sorted(str(value).casefold() for value in (arguments.get("exclude_domains") or ()))),
    )


def _equivalent_web_search(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> bool:
    """Conservatively match reformulations of one web-search intent."""

    current_query = current.get("query")
    previous_query = previous.get("query")
    if not isinstance(current_query, str) or not isinstance(previous_query, str):
        return False
    if _web_search_filters(current) != _web_search_filters(previous):
        return False

    current_words, current_numbers = _web_search_query_parts(current_query)
    previous_words, previous_numbers = _web_search_query_parts(previous_query)
    if current_numbers != previous_numbers or not current_words or not previous_words:
        return False

    return current_words == previous_words


def _completed_equivalent_web_search(
    arguments: dict[str, Any],
    executed: Sequence[ExecutedToolCall],
) -> ExecutedToolCall | None:
    """Find an earlier completed equivalent search in this run."""

    for call in reversed(executed):
        if (
            call.tool_name == "web_search"
            and call.result.ok
            and _equivalent_web_search(arguments, call.arguments)
        ):
            return call
    return None


def _duplicate_web_search_result(previous: ExecutedToolCall) -> ToolResult:
    """Return a model-facing hard rejection without calling a search provider."""

    previous_query = str(previous.arguments.get("query") or "").strip()
    previous_results = (
        previous.result.data.get("results") if isinstance(previous.result.data, dict) else None
    )
    next_action = (
        "Use the previously returned evidence"
        if previous_results
        else "The previous search returned no evidence; do not reformulate it"
    )
    error = (
        "An equivalent web search was already completed"
        f" (previous query: {previous_query!r}). Repeated or reformulated searches for the same "
        f"intent are blocked. {next_action}; call another appropriate tool for a distinct "
        "unresolved task, or produce the final answer."
    )
    observation = render_tool_error_observation(ToolErrorCode.DUPLICATE_CALL, error)
    return ToolResult(
        tool_name="web_search",
        ok=False,
        error=error,
        error_code=ToolErrorCode.DUPLICATE_CALL,
        retryable=False,
        metrics=ToolMetrics(
            latency_ms=0,
            response_bytes=len(observation.encode("utf-8")),
            model_tokens_estimate=max(1, len(observation) // 4),
            success=False,
            has_non_empty_answer=False,
        ),
    )


def _empty_result_follow_up(
    tool_name: str,
    available_tool_names: Sequence[str],
) -> dict[str, Any]:
    """Return structured guidance after no-results without mandating a fallback tool."""

    tool_example = ""
    if tool_name != "web_search" and "web_search" in available_tool_names:
        tool_example = " For example, use web_search when web evidence can answer the request."
    return {
        "status": "NO_RESULTS",
        "message": "The search completed successfully but returned no results.",
        "agent_guidance": {
            "retry_policy": (
                "Do not repeat an equivalent search using the same tool with the same search "
                "intent and materially equivalent parameters."
            ),
            "next_action": (
                "Choose another appropriate available tool or strategy, or conclude in the user's "
                f"language that no suitable result was found.{tool_example}"
            ),
        },
    }


# Kept out of the base system prompt so requests that do not discover places do
# not pay for it. It is appended to the corresponding tool observation: some
# OpenAI-compatible providers only permit a system message at the beginning of
# the conversation.
_PLACES_SEARCH_FOLLOW_UP = {
    "message": "The places search completed successfully and returned results.",
    "agent_guidance": {
        "retry_policy": (
            "Do not call places_search again with the same search intent and materially "
            "equivalent parameters. After a successful resolve of candidates supplied by "
            "web_search for one criterion, do not call places_search again for that criterion "
            "with additional or different candidates."
        ),
        "next_action": (
            "Use the returned results for the part of the user's request they answer. Before "
            "including a resolved candidate, verify that its place type or category matches the "
            "entity type requested. If a previous web_search supplied candidates for this part, "
            "do not follow resolve with places_search mode=area or mode=near to replace, expand, "
            "or pad that candidate set, and do not make another resolve call with more candidates. "
            "Return the verified subset even if fewer places qualify. If a distinct part remains "
            "unresolved and requires "
            "materially different search parameters, call places_search for that part; otherwise "
            "produce the final answer."
        ),
    },
}

_ROUTING_TOOL_FOLLOW_UP = {
    "message": "The routing calculation completed successfully and returned results.",
    "agent_guidance": {
        "retry_policy": (
            "Do not call routing_tool again with the same routing intent and materially "
            "equivalent parameters."
        ),
        "next_action": (
            "Use the returned routing results for the part of the user's request they answer. "
            "If a distinct routing task remains unresolved and requires materially different "
            "parameters, call routing_tool for that task; otherwise produce the final answer."
        ),
        "presentation_policy": (
            "For turn-by-turn directions, include every non-arrival step with its authoritative "
            "length_m. Do not use a rounded distance embedded in instruction. If street_name is "
            "absent, describe an unnamed road or path without inventing a name. Omit distance "
            "only for a zero-length arrival step."
        ),
    },
}


def _history_messages(history: Sequence[ConversationTurn]) -> list[LLMMessage]:
    """Stored turns -> model messages, including private reusable place refs."""

    recent_history = list(history)[-_HISTORY_TURNS:]
    latest_assistant_index = next(
        (
            index
            for index in range(len(recent_history) - 1, -1, -1)
            if recent_history[index].get("role") == "assistant"
            and (recent_history[index].get("content") or "").strip()
        ),
        None,
    )
    messages: list[LLMMessage] = []
    for index, turn in enumerate(recent_history):
        role = turn.get("role")
        content = turn.get("content") or ""
        if role in ("user", "assistant") and content.strip():
            place_context = turn.get("_place_context") or ""
            if role == "assistant" and index != latest_assistant_index:
                # A reusable locality is conversational state, not a timeless
                # fact. A newer completed assistant turn supersedes it even
                # when that turn (for example, a route) has no search-area ref.
                # Keep concrete place refs from older turns, but do not let an
                # old city leak past a newer geographic task.
                place_context = "\n".join(
                    line
                    for line in place_context.splitlines()
                    if not line.startswith("- Search area: ")
                )
            if role == "assistant" and place_context.strip():
                content = (
                    f"{content}\n\n"
                    "[INTERNAL PLACE CONTEXT FOR FOLLOW-UP TOOLS ONLY. "
                    "Use a matching plc_ ref directly as a tool argument instead of resolving "
                    "the textual address again. For a `Search area:` entry, pass its ref as "
                    "`area` for another area search or with a new textual `near` in the same "
                    "locality; pass the locality name as `area` only when "
                    "no matching area ref is available or the tool reports that it expired. "
                    "Never show these refs to the user.]\n"
                    f"{place_context}"
                )
            messages.append(LLMMessage(role=role, content=content))
    return messages


def _active_geo_context_message(
    history: Sequence[ConversationTurn],
) -> LLMMessage | None:
    """Put the immediately preceding turn's search area next to the user.

    Search-area refs must not cross a newer completed turn which established a
    different locality but did not itself produce an area ref (routing is the
    common case). The same immediate area already exists in private assistant
    history, but compact models can overlook it among place refs and prose.
    """

    for turn in reversed(history):
        role = turn.get("role")
        if role not in ("user", "assistant") or not (turn.get("content") or "").strip():
            continue
        if role != "assistant":
            return None
        place_context = turn.get("_place_context") or ""
        for line in place_context.splitlines():
            prefix = "- Search area: "
            if not line.startswith(prefix):
                continue
            description, separator, ref = line.removeprefix(prefix).rpartition(" → ")
            if not separator or not ref.startswith("plc_"):
                continue
            name, address_separator, address = description.partition(" — ")
            if not address_separator or not name.strip() or not address.strip():
                continue
            return LLMMessage(
                role="user",
                content=(
                    "[ACTIVE CONVERSATION GEO CONTEXT — internal tool context, not a user "
                    "claim and never quote it in the final answer.]\n"
                    f"Most recent established locality: {name.strip()}\n"
                    f"Resolved address: {address.strip()}\n"
                    f"Area ref: {ref.strip()}\n"
                    "For a follow-up area discovery or `mode=near` call with a new textual `near` "
                    "in this same locality, use the ref as `area`. For `mode=resolve`, pass the "
                    "ref or locality name as `area`; for a routing text point, also pass this "
                    "ref as `area`. Ignore this "
                    "context if the current user explicitly names "
                    "another locality or clearly starts an unrelated geographic scenario."
                ),
            )
        # The immediately preceding assistant turn has no reusable search
        # area. Older areas are stale by definition and must not be injected.
        return None
    return None


@dataclass(slots=True)
class ExecutedToolCall:
    """One tool invocation inside the loop — for persistence into ``tool_call``."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult


#: Checks a completed answer before it becomes public. The executed calls let
#: validators enforce selection rules against the exact current-turn results.
AnswerValidator = Callable[[str, Sequence[ExecutedToolCall]], Awaitable[str | None]]


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """Metrics for one physical call to the configured LLM provider."""

    turn_index: int
    model: str
    mode: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None


class OrchestratorExecutionError(RuntimeError):
    """Raised with partial observability data when the ReAct loop fails."""

    def __init__(
        self,
        cause: Exception,
        *,
        llm_calls: list[LLMCallRecord],
        tool_calls: list[ExecutedToolCall],
        turns: int,
        regenerations: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.llm_calls = llm_calls
        self.tool_calls = tool_calls
        self.turns = turns
        self.regenerations = regenerations


@dataclass(slots=True)
class OrchestratorRun:
    """The whole loop's outcome, shaped so the pipeline persists it as before."""

    #: Accumulated across turns: full trace, final content, summed usage.
    llm_response: LLMResponse
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)

    @property
    def llm_turns(self) -> int:
        """How many times the model was called (>=1)."""
        return self._turns

    @property
    def regenerations(self) -> int:
        """How many times a rejected answer was sent back to be rewritten."""
        return self._regenerations

    _turns: int = 1
    _regenerations: int = 0
    llm_call_latencies_ms: list[int] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    #: Both the original empty step and its existing in-run recovery were empty.
    #: The outer pipeline uses this to trigger one clean orchestration retry.
    empty_response: bool = False

    @property
    def llm_latency_ms(self) -> int:
        return sum(self.llm_call_latencies_ms)

    @property
    def tool_latency_ms(self) -> int:
        return sum(
            call.result.metrics.latency_ms
            for call in self.tool_calls
            if call.result.metrics is not None
        )

    @property
    def upstream_call_count(self) -> int:
        return sum(
            len(call.result.metrics.upstream_calls)
            for call in self.tool_calls
            if call.result.metrics is not None
        )


def to_openai_tools(specs: Sequence[ToolSpec[Any, Any]]) -> list[dict[str, Any]]:
    """Tool specs -> OpenAI function-calling schema handed to the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                # The full Pydantic schema is the backend validator. A tool may
                # expose a smaller equivalent contract to the model, avoiding
                # repeated validation-only detail in every ReAct prompt.
                "parameters": (
                    spec.llm_parameters
                    if spec.llm_parameters is not None
                    else spec.input_model.model_json_schema()
                ),
            },
        }
        for spec in specs
    ]


def _observation_text(
    result: ToolResult,
    follow_up: dict[str, Any] | None = None,
) -> str:
    """Render a tool result for the model. Errors are data, not exceptions."""
    if result.ok:
        data = _model_observation_data(result)
        if follow_up is not None:
            # Guidance belongs to this specific tool result and stays inside the
            # same JSON function-call output rather than becoming a new message.
            data = {**data, **follow_up}
        return render_tool_observation(
            data,
            result.warnings,
        )
    assert result.error_code is not None
    return render_tool_error_observation(
        result.error_code,
        result.error or "Unknown tool error",
        max_chars=_MAX_TOOL_ERROR_CHARS,
    )


def _model_observation_data(result: ToolResult) -> dict[str, Any]:
    """Return the model-facing view of a tool result.

    Tool results remain complete in ``ToolResult.data`` for persistence, audit,
    map expansion, and follow-up backend work.  This projection exists only to
    avoid spending prompt tokens on provider ids and empty optional fields.
    """

    data = result.data or {}
    if result.tool_name != "places_search":
        return data

    try:
        output = PlacesSearchOutput.model_validate(data)
    except ValidationError:
        # A custom or historical implementation may not satisfy the runtime
        # places contract. Preserve its original observation rather than hiding
        # data the model might need to recover from it.
        return data

    payload: dict[str, Any] = {
        "places": [_place_observation(place) for place in output.places],
    }
    if output.area is not None:
        payload["area"] = output.area.model_dump(mode="json")
    if output.truncated:
        payload["truncated"] = True
    if output.anchor is not None:
        payload["anchor"] = output.anchor
    if output.resolved:
        payload["resolved"] = [_resolution_observation(item) for item in output.resolved]
    return payload


def _place_observation(place: Place) -> dict[str, Any]:
    """Keep all user-relevant place facts while dropping provider-only noise."""

    payload: dict[str, Any] = {
        "ref": place.ref,
        "name": place.name,
        "address": place.address,
    }
    if place.categories:
        payload["categories"] = place.categories
    if place.phones:
        payload["phones"] = place.phones
    if place.rating is not None:
        payload["rating"] = place.rating
    if place.review_count is not None:
        payload["review_count"] = place.review_count
    if place.hours_text is not None:
        payload["hours"] = place.hours_text
    if place.open_24h is True:
        payload["open_24h"] = True
    if place.is_open_now is not None:
        payload["is_open_now"] = place.is_open_now
    if place.accessibility:
        payload["accessibility"] = place.accessibility
    if place.distance_m is not None:
        payload["distance_m"] = place.distance_m
    return payload


def _resolution_observation(resolution: OrganisationResolution) -> dict[str, Any]:
    """Compact one ``mode=resolve`` outcome without losing usable refs."""

    payload: dict[str, Any] = {
        "client_id": resolution.client_id,
        "input_name": resolution.input_name,
        "status": resolution.status,
    }
    if resolution.place is not None:
        payload["place"] = _place_observation(resolution.place)
    if resolution.options:
        payload["options"] = [_place_observation(place) for place in resolution.options]
    if resolution.error is not None:
        payload["error"] = resolution.error
    return payload


def _final_step(turn: Sequence[ReActStep]) -> ReActStep | None:
    for step in turn:
        if step.type is ReActStepType.FINAL_ANSWER:
            return step
    return None


def _actions(turn: Sequence[ReActStep]) -> list[ReActStep]:
    """Every tool the model asked for this turn — models do emit parallel calls."""
    return [step for step in turn if step.type is ReActStepType.ACTION and step.tool_name]


def _sum_usage(a: LLMUsage, b: LLMUsage) -> LLMUsage:
    return LLMUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
    )


_EXPOSED_REASONING_PATTERNS = (
    re.compile(
        r"(?im)^\s*(?:i have gathered|i have resolved|"
        r"i will (?:write|draft|check|ensure|structure|add|make)|"
        r"let(?:'s| us) draft|draft:|final check|all good)\b"
    ),
    re.compile(
        r"(?im)^\s*(?:я (?:собрал|наш[её]л|проверю|напишу|составлю|добавлю)|"
        r"давайте составим|черновик:|финальная проверка)\b"
    ),
)


def _exposed_reasoning_problem(answer: str) -> str | None:
    """Reject obvious planning/checklist text accidentally emitted as an answer.

    A single first-person phrase may be ordinary user-facing prose. Multiple
    planning markers indicate that the model put its scratchpad in ``content``;
    OpenAI-compatible APIs provide no separate type marker for that case.
    """

    matches = sum(len(pattern.findall(answer)) for pattern in _EXPOSED_REASONING_PATTERNS)
    if matches < 2:
        return None
    return (
        "The previous response exposed internal planning instead of answering the user. "
        "Do not describe what you gathered, what you will write, drafts, checks, constraints, "
        "or reference validation. Start immediately with the concise user-facing Markdown "
        "answer, using the available observations and exact required refs."
    )


class Orchestrator:
    """Drives one request's ReAct loop. Stateless across requests."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        tool_executor: ToolExecutor,
        tool_specs: Sequence[ToolSpec[Any, Any]] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: str | None = None,
        echo_store: EchoStore | None = None,
        answer_validator: AnswerValidator | None = None,
        max_regenerations: int = DEFAULT_MAX_REGENERATIONS,
        progress_callback: ProgressCallback | None = None,
        answer_chunk_callback: AnswerChunkCallback | None = None,
    ) -> None:
        self._llm = llm
        self._executor = tool_executor
        self._tools = to_openai_tools(tool_specs)
        self._tool_names = tuple(spec.name for spec in tool_specs)
        self._max_steps = max(1, max_steps)
        self._temperature = temperature
        self._top_p = top_p
        # Describe exactly the tools we advertise: a documented-but-uncallable
        # tool just invites a wasted turn on "unknown tool".
        self._system_prompt = system_prompt or build_orchestrator_prompt(self._tool_names)
        self._echo_store = echo_store
        self._answer_validator = answer_validator
        self._max_regenerations = max(0, max_regenerations)
        self._progress_callback = progress_callback
        self._answer_chunk_callback = answer_chunk_callback

    def _report_progress(self, kind: str, **details: Any) -> None:
        """Report best-effort UI progress without affecting the agent run."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(kind, details)
        except Exception:
            logger.warning("progress_callback_failed kind=%s", kind, exc_info=True)

    async def _validate(
        self,
        answer: str,
        executed: Sequence[ExecutedToolCall],
    ) -> str | None:
        """Run an optional final-answer validator without risking the answer."""

        if problem := _exposed_reasoning_problem(answer):
            return problem
        if self._answer_validator is None:
            return None
        try:
            return await self._answer_validator(answer, executed)
        except Exception:
            logger.warning("answer_validator_failed", exc_info=True)
            return None

    def _unknown_tool_result(self, tool_name: str) -> ToolResult:
        """Reject an invented tool name and tell the model what it may call.

        The executor already refuses unknown tools, but its message does not list
        the alternatives — so the model is free to guess again. Naming the closed
        set lets it self-correct in one turn.
        """
        available = ", ".join(self._tool_names) or "(none configured)"
        error = f"unknown tool: {tool_name}. Available tools: {available}"
        observation = render_tool_error_observation(ToolErrorCode.INVALID_INPUT, error)
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=error,
            error_code=ToolErrorCode.INVALID_INPUT,
            retryable=False,
            metrics=ToolMetrics(
                latency_ms=0,
                response_bytes=len(observation.encode("utf-8")),
                model_tokens_estimate=max(1, len(observation) // 4),
                success=False,
                has_non_empty_answer=False,
            ),
        )

    async def _record_echo(self, result: ToolResult) -> None:
        """Map tool_hash -> payload, so a cited result can be proven after the fact."""
        if self._echo_store is None or not result.tool_hash:
            return
        try:
            await self._echo_store.set_echo(result.tool_hash, result.model_dump_json())
        except Exception:
            logger.warning("echo_write_failed tool_hash=%s", result.tool_hash, exc_info=True)

    async def run(
        self, user_message: str, *, history: Sequence[ConversationTurn] = ()
    ) -> OrchestratorRun:
        active_geo_context = _active_geo_context_message(history)
        messages = [
            LLMMessage(role="system", content=self._system_prompt),
            *_history_messages(history),
            *([active_geo_context] if active_geo_context is not None else []),
            LLMMessage(role="user", content=user_message),
        ]
        steps: list[ReActStep] = []
        executed: list[ExecutedToolCall] = []
        usage = LLMUsage()
        model = ""
        mode = "mock"
        final_answer = ""
        turns = 0
        regenerations = 0
        llm_call_latencies_ms: list[int] = []
        llm_calls: list[LLMCallRecord] = []
        places_search_instruction_added = False
        empty_response_retries = 0
        validation_retries = 0
        empty_retry_instruction_pending = False
        empty_response = False
        run_started = perf_counter()

        for _ in range(self._max_steps):
            # Always reserve the last model turn for composing an answer from the
            # observations gathered so far. Without this, a model can spend every
            # turn on tools and never get a chance to answer the user.
            tools_for_turn = self._tools if turns < self._max_steps - 1 else ()
            if not tools_for_turn and self._tools and not empty_retry_instruction_pending:
                # Several OpenAI-compatible providers reject a system message
                # anywhere but the first position. This internal control message
                # is valid as a user turn after the preceding tool observations.
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "Tool-call budget reached. Using the available observations, "
                            "produce the final answer now; do not request another tool."
                        ),
                    )
                )
            self._report_progress("model_started", turn_index=turns + 1)
            llm_started = perf_counter()
            try:
                llm_request = LLMRequest(
                    messages=messages,
                    tools=tools_for_turn,
                    temperature=self._temperature,
                    top_p=self._top_p,
                )
                streamed_text = False
                public_stream = InternalRefStreamFilter()

                def emit_chunk(
                    chunk: str,
                    stream: InternalRefStreamFilter = public_stream,
                ) -> None:
                    nonlocal streamed_text
                    if not chunk or self._answer_chunk_callback is None:
                        return
                    for public_chunk in stream.feed(chunk):
                        if public_chunk:
                            streamed_text = True
                            self._answer_chunk_callback(public_chunk)

                response = await self._llm.generate_stream(llm_request, emit_chunk)
                empty_retry_instruction_pending = False
                if self._answer_chunk_callback is not None:
                    for public_chunk in public_stream.finish():
                        if public_chunk:
                            streamed_text = True
                            self._answer_chunk_callback(public_chunk)
            except Exception as exc:
                llm_latency_ms = int((perf_counter() - llm_started) * 1000)
                turns += 1
                llm_calls.append(
                    LLMCallRecord(
                        turn_index=turns,
                        model="unknown",
                        mode="unknown",
                        latency_ms=llm_latency_ms,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        success=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                logger.exception("llm_turn_failed turn=%s latency_ms=%s", turns, llm_latency_ms)
                raise OrchestratorExecutionError(
                    exc,
                    llm_calls=llm_calls,
                    tool_calls=executed,
                    turns=turns,
                    regenerations=regenerations,
                ) from exc
            llm_latency_ms = int((perf_counter() - llm_started) * 1000)
            llm_call_latencies_ms.append(llm_latency_ms)
            turns += 1
            llm_calls.append(
                LLMCallRecord(
                    turn_index=turns,
                    model=response.model,
                    mode=response.mode,
                    latency_ms=llm_latency_ms,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    reasoning_tokens=response.usage.reasoning_tokens,
                    finish_reason=response.finish_reason,
                )
            )
            logger.info(
                "llm_turn_complete turn=%s model=%s mode=%s latency_ms=%s"
                " prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s"
                " visible_completion_tokens=%s total_tokens=%s finish_reason=%s",
                turns,
                response.model,
                response.mode,
                llm_latency_ms,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.reasoning_tokens,
                max(0, response.usage.completion_tokens - response.usage.reasoning_tokens),
                response.usage.total_tokens,
                response.finish_reason,
            )
            model, mode = response.model, response.mode
            usage = _sum_usage(usage, response.usage)
            turn = response.trace.steps
            steps.extend(turn)

            # A turn that produced a final answer ends the loop.
            final = _final_step(turn)
            final_candidate = (
                (final.content if final is not None else response.trace.final_answer)
                or response.content
                if final is not None or response.trace.final_answer
                else ""
            )
            if final_candidate.strip():
                problem = await self._validate(final_candidate, executed)
                if problem is not None and validation_retries < self._max_regenerations:
                    validation_retries += 1
                    regenerations += 1
                    if streamed_text:
                        self._report_progress("answer_reset", turn_index=turns)
                    self._report_progress(
                        "answer_validation",
                        turn_index=turns,
                        regeneration=regenerations,
                        reason="invalid_refs",
                    )
                    logger.info(
                        "answer_rejected_regenerating attempt=%s reason=%s",
                        validation_retries,
                        problem,
                    )
                    messages.append(LLMMessage(role="user", content=problem))
                    continue
                final_answer = final_candidate
                self._report_progress("answer_ready", turn_index=turns)
                break

            actions = _actions(turn)
            if actions and streamed_text:
                # Text accompanying a tool call is narration, not the answer.
                self._report_progress("answer_reset", turn_index=turns)
            if not actions:
                # A provider can exhaust its completion budget on hidden reasoning
                # and return neither text nor a tool call. Never persist that as a
                # successful empty answer: give it one explicit recovery turn.
                if response.content.strip():
                    final_answer = response.content
                    self._report_progress("answer_ready", turn_index=turns)
                    break
                if empty_response_retries == 0 and turns < self._max_steps:
                    empty_response_retries += 1
                    regenerations += 1
                    empty_retry_instruction_pending = True
                    next_turn_has_tools = turns < self._max_steps - 1
                    if executed:
                        instruction = (
                            "The previous generation was empty. Stop reasoning now and do not "
                            "continue the analysis. Using the available tool observations, "
                            "immediately write a concise, non-empty user-facing final answer."
                        )
                    elif next_turn_has_tools:
                        instruction = (
                            "The previous generation was empty. Stop extended reasoning now. "
                            "Immediately call the required tool, or write a concise, non-empty "
                            "user-facing final answer if no tool is required."
                        )
                    else:
                        instruction = (
                            "The previous generation was empty. Stop reasoning now and "
                            "immediately write a concise, non-empty user-facing final answer; "
                            "no tool calls are available in this recovery turn."
                        )
                    messages.append(LLMMessage(role="user", content=instruction))
                    self._report_progress(
                        "answer_validation",
                        turn_index=turns,
                        regeneration=regenerations,
                        reason="empty_response",
                    )
                    logger.warning("empty_llm_response_retrying turn=%s", turns)
                    continue

                # The cheap recovery inside this ReAct run is exhausted. Leave
                # the answer empty so the pipeline can expose a waiting message
                # and start a new run with clean orchestration state.
                empty_response = True
                logger.warning("empty_llm_response_recovery_exhausted turn=%s", turns)
                break

            if not tools_for_turn:
                # A provider ignored the no-tools final turn. Do not execute an
                # extra call and leave no opportunity to answer.
                final_answer = "Не удалось сформировать ответ по полученным данным."
                steps.append(ReActStep(type=ReActStepType.FINAL_ANSWER, content=final_answer))
                self._report_progress("answer_ready", turn_index=turns, step_budget_exhausted=True)
                break

            # Index of the turn's first step inside the accumulated trace, so each
            # executed call can point back at its own ACTION step.
            turn_base = len(steps) - len(turn)
            tool_calls_payload: list[dict[str, Any]] = []
            tool_messages: list[LLMMessage] = []

            for offset, action in enumerate(turn):
                if action not in actions:
                    continue
                tool_name = action.tool_name or ""
                arguments = dict(action.tool_input or {})
                if tool_name == "places_search":
                    corrected_arguments = _restore_russian_generic_places_query(
                        arguments,
                        user_message,
                    )
                    if corrected_arguments is not arguments:
                        logger.info(
                            "places_search_query_language_restored original_query=%r "
                            "corrected_query=%r category=%r",
                            arguments.get("query"),
                            corrected_arguments.get("query"),
                            arguments.get("category"),
                        )
                        arguments = corrected_arguments
                # Prefer the provider's own id so the reply matches its call.
                call_id = action.tool_call_id or f"call_{len(executed)}"

                self._report_progress(
                    "tool_started",
                    turn_index=turns,
                    tool_name=tool_name,
                    tool_input=arguments,
                )

                # Closed set: an invented name never reaches the executor, and the
                # model is told which names are real.
                if self._tool_names and tool_name not in self._tool_names:
                    result = self._unknown_tool_result(tool_name)
                elif (
                    tool_name == "web_search"
                    and (previous_search := _completed_equivalent_web_search(arguments, executed))
                    is not None
                ):
                    result = _duplicate_web_search_result(previous_search)
                    logger.info(
                        "equivalent_web_search_blocked previous_step=%s previous_query=%r "
                        "new_query=%r",
                        previous_search.step_index,
                        previous_search.arguments.get("query"),
                        arguments.get("query"),
                    )
                else:
                    result = await self._executor.run(tool_name, arguments)

                follow_up: dict[str, Any] | None = None
                has_empty_answer = (
                    result.ok
                    and result.metrics is not None
                    and not result.metrics.has_non_empty_answer
                )
                if has_empty_answer:
                    follow_up = _empty_result_follow_up(
                        tool_name,
                        self._tool_names,
                    )
                elif (
                    tool_name == "web_search"
                    and result.ok
                    and isinstance(result.data, dict)
                    and bool(result.data.get("results"))
                ):
                    follow_up = _WEB_RESULTS_FOLLOW_UP
                elif (
                    tool_name == "places_search"
                    and result.ok
                    and not places_search_instruction_added
                ):
                    follow_up = _PLACES_SEARCH_FOLLOW_UP
                    places_search_instruction_added = True
                elif tool_name == "routing_tool" and result.ok:
                    follow_up = _ROUTING_TOOL_FOLLOW_UP
                await self._record_echo(result)

                self._report_progress(
                    "tool_finished",
                    turn_index=turns,
                    tool_name=tool_name,
                    ok=result.ok,
                )

                logger.info(
                    "orchestrator_tool_complete turn=%s tool_name=%s ok=%s"
                    "latency_ms=%s upstream_calls=%s upstream_latency_ms=%s",
                    turns,
                    tool_name,
                    result.ok,
                    result.metrics.latency_ms if result.metrics else None,
                    len(result.metrics.upstream_calls) if result.metrics else 0,
                    result.metrics.upstream_latency_ms if result.metrics else None,
                )

                executed.append(
                    ExecutedToolCall(
                        step_index=turn_base + offset,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result,
                    )
                )
                observation = _observation_text(result, follow_up)
                steps.append(
                    ReActStep(
                        type=ReActStepType.OBSERVATION,
                        content=observation,
                        tool_name=tool_name,
                        tool_input=arguments,
                        tool_call_id=call_id,
                    )
                )
                tool_calls_payload.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                )
                tool_messages.append(
                    LLMMessage(role="tool", content=observation, tool_call_id=call_id)
                )

            # Replay the turn in the shape tool-calling models are trained on:
            # the assistant's tool_calls, then one tool message per result.
            messages.append(LLMMessage(role="assistant", content="", tool_calls=tool_calls_payload))
            messages.extend(tool_messages)
        else:
            # Step budget exhausted without a final answer.
            final_answer = "Не удалось завершить запрос в отведённое число шагов."
            steps.append(ReActStep(type=ReActStepType.FINAL_ANSWER, content=final_answer))
            self._report_progress("answer_ready", turn_index=turns, step_budget_exhausted=True)

        trace = ReActTrace(steps=steps, final_answer=final_answer)
        llm_response = LLMResponse(
            model=model or "geoagent-model",
            mode=mode,
            content=final_answer,
            trace=trace,
            usage=usage,
        )
        logger.info(
            "orchestrator_complete turns=%s tool_calls=%s regenerations=%s"
            "llm_latency_ms=%s total_latency_ms=%s",
            turns,
            len(executed),
            regenerations,
            sum(llm_call_latencies_ms),
            int((perf_counter() - run_started) * 1000),
        )
        return OrchestratorRun(
            llm_response=llm_response,
            tool_calls=executed,
            _turns=turns,
            _regenerations=regenerations,
            llm_call_latencies_ms=llm_call_latencies_ms,
            llm_calls=llm_calls,
            empty_response=empty_response,
        )
