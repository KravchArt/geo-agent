"""Phase 0 request pipeline with parallel preflight gates.

UI -> FastAPI -> request log -> (scope gate || censorship gate) -> main LLM when
allowed -> Redis session state + Postgres observability -> response.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import Settings, get_settings
from backend.app.db.models import (
    GateCheckLog,
    LLMCallMetric,
    Metrics,
    ModelResponse,
    PipelineStageMetric,
    ReactTrace,
    Request,
    ToolCall,
    UpstreamCallMetric,
)
from backend.app.llm.base import get_llm_client
from backend.app.observability import RequestMetricsCollector
from backend.app.redis.client import RedisClient
from backend.app.services.conversations import ConversationStore
from backend.app.services.gates import (
    ClassifierScopeGate,
    GateEvaluator,
    SessionUnlockedScopeGate,
    evaluate_output_censorship,
    evaluate_preflight_gates,
)
from backend.app.services.grounding import GroundingReport, verify_answer_refs
from backend.app.services.map_places import (
    load_map_places,
    missing_listed_place_names,
    reusable_search_areas,
    selected_eligible_place_refs,
    selected_search_anchor_refs,
)
from backend.app.services.orchestrator import (
    ExecutedToolCall,
    LLMCallRecord,
    Orchestrator,
    OrchestratorExecutionError,
    OrchestratorRun,
)
from backend.app.services.user_context import (
    CityReverseGeocoder,
    append_user_context,
    resolve_user_context,
)
from common.models import (
    AgentRequest,
    AgentResponse,
    GateDecision,
    GateResults,
    LLMResponse,
    LLMUsage,
    MapData,
    MapPlace,
    ReActStep,
    ReActStepType,
    ReActTrace,
    SourceCitation,
)
from tools.base import ToolSpec
from tools.executor import ToolExecutor
from tools.geo.place_store import PlaceStore
from tools.geo.places_search.schemas import ResolvedSearchArea
from tools.web.source_store import SourceStore


def _missing_required_web_citation(
    report: GroundingReport,
    executed_calls: Sequence[ExecutedToolCall],
) -> bool:
    """Require a citation to evidence returned by this request's web search."""

    returned_refs = {
        ref
        for executed in executed_calls
        if executed.tool_name == "web_search"
        and executed.result.ok
        and isinstance(executed.result.data, dict)
        for result in (executed.result.data.get("results") or [])
        if isinstance(result, dict) and isinstance((ref := result.get("ref")), str)
    }
    return bool(returned_refs) and returned_refs.isdisjoint(report.source_refs)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, object]], None]

_OUT_OF_SCOPE_ANSWER = "I can only help with travel and location-related questions."
_CENSORSHIP_ANSWER = "I cannot help with that request."
_EMPTY_RESPONSE_RETRY_MESSAGE = "I need a little more time."
_EMPTY_RESPONSE_FAILURE_MESSAGE = "I couldn't provide an answer."
_INTERNAL_REF_LINK_RE = re.compile(r"\s*\[[^\]\n]+\]\((?:src|plc)_[a-z0-9]+\)", re.IGNORECASE)
_INTERNAL_REF_BRACKETS_RE = re.compile(r"\s*\[\[?(?:src|plc)_[a-z0-9]+\]?\]", re.IGNORECASE)
_INTERNAL_SOURCES_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:\*{1,2})?источники(?:\*{1,2})?\s*:\s*[,;·\s]*(?:\*{1,2})?$"
)


def _decision_payload(gates: GateResults) -> dict[str, object]:
    return gates.model_dump(mode="json")


async def _cited_web_sources(
    grounding: GroundingReport | None,
    source_store: SourceStore | None,
    executed_calls: Sequence[ExecutedToolCall] = (),
) -> list[SourceCitation]:
    """Expand cited sources followed by every result returned by web_search."""

    if source_store is None:
        return []

    refs = list(grounding.source_refs if grounding is not None else [])
    for executed in executed_calls:
        if (
            executed.tool_name != "web_search"
            or not executed.result.ok
            or not isinstance(executed.result.data, dict)
        ):
            continue
        results = executed.result.data.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            ref = result.get("ref")
            if isinstance(ref, str) and ref not in refs:
                refs.append(ref)

    citations: list[SourceCitation] = []
    for ref in refs:
        record = await source_store.get(ref)
        if record is not None:
            citations.append(
                SourceCitation(
                    ref=record.ref,
                    title=record.title,
                    url=record.url,
                    domain=record.domain,
                    published_date=record.published_date,
                    snippet=record.snippet,
                )
            )
    return citations


def _hide_internal_refs(answer: str) -> str:
    """Remove model-facing refs after they have been expanded for the UI."""
    cleaned = _INTERNAL_REF_LINK_RE.sub("", answer)
    cleaned = _INTERNAL_REF_BRACKETS_RE.sub("", cleaned)
    cleaned = _INTERNAL_SOURCES_LINE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _outcome(
    gates: GateResults,
) -> tuple[
    Literal["completed", "rejected"],
    Literal["out_of_scope", "censorship", "output_censorship"] | None,
    str,
]:
    """Apply deterministic precedence when both gates reject.

    Censorship wins over scope because it is the safety boundary and should not be
    obscured by a simultaneous out-of-scope result.
    """
    if not gates.censorship.passed:
        return "rejected", "censorship", _CENSORSHIP_ANSWER
    if not gates.scope.passed:
        return "rejected", "out_of_scope", _OUT_OF_SCOPE_ANSWER
    return "completed", None, ""


def _combine_orchestrator_runs(runs: Sequence[OrchestratorRun]) -> OrchestratorRun:
    """Preserve trace, calls, latency and token cost across a clean retry."""
    if not runs:
        raise ValueError("at least one orchestrator run is required")
    final_run = runs[-1]
    steps: list[ReActStep] = []
    tool_calls: list[ExecutedToolCall] = []
    llm_calls: list[LLMCallRecord] = []
    usage = LLMUsage()
    turn_offset = 0
    for run in runs:
        step_offset = len(steps)
        steps.extend(run.llm_response.trace.steps)
        tool_calls.extend(
            replace(call, step_index=call.step_index + step_offset) for call in run.tool_calls
        )
        llm_calls.extend(
            replace(call, turn_index=call.turn_index + turn_offset) for call in run.llm_calls
        )
        turn_offset += run.llm_turns
        current = run.llm_response.usage
        usage = LLMUsage(
            prompt_tokens=usage.prompt_tokens + current.prompt_tokens,
            completion_tokens=usage.completion_tokens + current.completion_tokens,
            total_tokens=usage.total_tokens + current.total_tokens,
            reasoning_tokens=usage.reasoning_tokens + current.reasoning_tokens,
        )
    response = final_run.llm_response.model_copy(
        update={
            "trace": ReActTrace(steps=steps, final_answer=final_run.llm_response.content),
            "usage": usage,
        }
    )
    return OrchestratorRun(
        llm_response=response,
        tool_calls=tool_calls,
        _turns=turn_offset,
        _regenerations=sum(run.regenerations for run in runs) + len(runs) - 1,
        llm_call_latencies_ms=[latency for run in runs for latency in run.llm_call_latencies_ms],
        llm_calls=llm_calls,
        empty_response=final_run.empty_response,
    )


def _failed_orchestrator_run(error: OrchestratorExecutionError) -> OrchestratorRun:
    """Represent a timed-out attempt so a successful retry keeps its costs."""
    usage = LLMUsage(
        prompt_tokens=sum(call.prompt_tokens for call in error.llm_calls),
        completion_tokens=sum(call.completion_tokens for call in error.llm_calls),
        total_tokens=sum(call.total_tokens for call in error.llm_calls),
        reasoning_tokens=sum(call.reasoning_tokens for call in error.llm_calls),
    )
    response = LLMResponse(
        model=error.llm_calls[-1].model if error.llm_calls else "unknown",
        mode="vllm",
        content="",
        trace=ReActTrace(),
        usage=usage,
    )
    return OrchestratorRun(
        llm_response=response,
        tool_calls=list(error.tool_calls),
        _turns=error.turns,
        _regenerations=error.regenerations,
        llm_call_latencies_ms=[call.latency_ms for call in error.llm_calls],
        llm_calls=list(error.llm_calls),
    )


def _is_llm_timeout(error: OrchestratorExecutionError) -> bool:
    """Only an actual provider deadline, not every orchestration error, is retryable."""
    return isinstance(error.cause, TimeoutError)


def _gate_attempts(decision: GateDecision) -> list[GateDecision]:
    """Return physical attempts for a gate, or the decision itself for simple gates."""
    return list(decision.attempts) if decision.attempts else [decision]


def _all_gate_attempts(gates: GateResults) -> list[tuple[GateDecision, str]]:
    decisions: list[tuple[GateDecision, str]] = []
    decisions.extend((decision, "input") for decision in _gate_attempts(gates.scope))
    decisions.extend((decision, "input") for decision in _gate_attempts(gates.censorship))
    if gates.output_censorship is not None:
        decisions.extend(
            (decision, "output") for decision in _gate_attempts(gates.output_censorship)
        )
    return decisions


def _persist_gate_logs(*, db: AsyncSession, request_row: Request, gates: GateResults) -> None:
    per_gate_index: dict[tuple[str, str], int] = {}
    for decision, phase in _all_gate_attempts(gates):
        key = (decision.name.value, phase)
        per_gate_index[key] = per_gate_index.get(key, 0) + 1
        db.add(
            GateCheckLog(
                request_id=request_row.id,
                gate_name=decision.name.value,
                phase=phase,
                call_index=per_gate_index[key],
                provider=decision.provider,
                model_name=decision.model,
                verdict=decision.verdict.value,
                passed=decision.passed,
                reason=decision.reason,
                matched_rules={"rules": decision.matched_rules},
                confidence=decision.confidence,
                latency_ms=decision.latency_ms,
                success=decision.success,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
                total_tokens=decision.total_tokens,
                error_type=decision.error_type,
                error_message=decision.error_message,
                raw_response=decision.model_dump(mode="json"),
            )
        )


def _log_gate_decision(request_id: object, decision: GateDecision, phase: str) -> None:
    logger.info(
        "gate_call_complete request_id=%s gate=%s phase=%s provider=%s model=%s "
        "latency_ms=%s passed=%s success=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        request_id,
        decision.name.value,
        phase,
        decision.provider,
        decision.model,
        decision.latency_ms,
        decision.passed,
        decision.success,
        decision.prompt_tokens,
        decision.completion_tokens,
        decision.total_tokens,
    )


#: Redis session field holding the "this session already passed scope" flag.
SCOPE_UNLOCKED_FIELD = "scope_unlocked"


def _session_unlock_applies(settings: Settings, scope_gate: GateEvaluator | None) -> bool:
    """The unlock only exists to paper over the classifier being contextless.

    With any other scope gate it would be a pure loss: the LLM scoper and the regex
    gate both read history already, so re-checking costs them nothing extra and
    keeps later turns filtered.
    """
    return settings.scope_session_unlock and isinstance(scope_gate, ClassifierScopeGate)


async def _scope_is_unlocked(
    *,
    redis: RedisClient,
    settings: Settings,
    session_id: str,
    scope_gate: GateEvaluator | None,
) -> bool:
    if not _session_unlock_applies(settings, scope_gate):
        return False
    try:
        return await redis.get_session_value(session_id, SCOPE_UNLOCKED_FIELD) == "1"
    except Exception:
        # Redis being unavailable must not skip the gate — fail closed and classify.
        logger.warning("scope_unlock_read_failed session_id=%s", session_id, exc_info=True)
        return False


async def _remember_scope_verdict(
    *,
    redis: RedisClient,
    settings: Settings,
    session_id: str,
    scope: GateDecision,
    scope_gate: GateEvaluator | None,
) -> None:
    """Record the first in-scope verdict so later turns skip the classifier."""
    if not scope.passed or not _session_unlock_applies(settings, scope_gate):
        return
    try:
        await redis.set_session_value(session_id, SCOPE_UNLOCKED_FIELD, "1")
    except Exception:
        # Losing the flag only costs one extra classification next turn.
        logger.warning("scope_unlock_write_failed session_id=%s", session_id, exc_info=True)


async def _cache_pipeline_state(
    *,
    redis: RedisClient,
    payload: AgentRequest,
    gates: GateResults,
    status: str,
    answer: str,
) -> None:
    await redis.set_session_value(payload.session_id, "last_user_message", payload.message)
    await redis.set_session_value(
        payload.session_id,
        "last_gate_results",
        json.dumps(_decision_payload(gates), ensure_ascii=False),
    )
    await redis.set_session_value(payload.session_id, "last_pipeline_status", status)
    await redis.set_session_value(payload.session_id, "last_answer", answer)


async def run_pipeline(
    payload: AgentRequest,
    *,
    db: AsyncSession,
    redis: RedisClient,
    censorship_gate: GateEvaluator | None = None,
    scope_gate: GateEvaluator | None = None,
    tool_executor: ToolExecutor | None = None,
    tool_specs: Sequence[ToolSpec[Any, Any]] = (),
    place_store: PlaceStore | None = None,
    source_store: SourceStore | None = None,
    user_location_reverse_geocoder: CityReverseGeocoder | None = None,
    progress_callback: ProgressCallback | None = None,
    client_id: str | None = None,
    conversation_store: ConversationStore | None = None,
) -> AgentResponse:
    """Execute one request and treat gate rejection as a successful business outcome."""
    durable_conversations = conversation_store or ConversationStore(db)
    # This is deliberately before request logging, gates and the LLM. A caller
    # cannot spend model/tool resources against somebody else's conversation.
    await durable_conversations.ensure_chat_access(payload.session_id, client_id)

    started = perf_counter()
    request_row = Request(
        session_id=payload.session_id,
        user_query=payload.message,
        status="running",
    )
    db.add(request_row)
    await db.flush()
    metrics = RequestMetricsCollector(str(request_row.id), payload.session_id, logger)
    gates: GateResults | None = None

    def report_progress(
        stage: str,
        message: str,
        *,
        status: Literal["running", "completed", "error"] = "running",
        **details: object,
    ) -> None:
        """Emit best-effort public progress; reporting must never break a request."""
        if progress_callback is None:
            return
        event: dict[str, object] = {
            "type": "progress",
            "stage": stage,
            "status": status,
            "message": message,
            **details,
        }
        try:
            progress_callback(event)
        except Exception:
            logger.warning("pipeline_progress_callback_failed stage=%s", stage, exc_info=True)

    report_progress("scope", "Checking request scope…")
    report_progress("censorship", "Checking request safety…")

    try:
        settings = get_settings()
        # Postgres is the sole conversation source of truth. In particular, do
        # not fall back to ownerless Redis state by session id: a guessed legacy
        # id or a delete/cache race must never resurrect transcript context.
        with metrics.stage("db_history_read"):
            history = await durable_conversations.recent_history(payload.session_id, limit=10)
        metrics.set_count("history_turns", len(history))
        with metrics.stage("user_context_resolution"):
            user_context = await resolve_user_context(
                payload.user_context,
                settings=settings,
                session_id=payload.session_id,
                place_store=place_store,
                reverse_geocoder=user_location_reverse_geocoder,
            )
        model_message = append_user_context(payload.message, user_context)
        # A session that already passed the scope check keeps its pass, because the
        # classifier is contextless and would reject its own follow-ups. Read the
        # flag before the gates so the skip is visible in the decision they return.
        scope_unlocked = await _scope_is_unlocked(
            redis=redis,
            settings=settings,
            session_id=payload.session_id,
            scope_gate=scope_gate,
        )
        gates = await evaluate_preflight_gates(
            payload.message,
            scope_gate=SessionUnlockedScopeGate() if scope_unlocked else scope_gate,
            censorship_gate=censorship_gate,
            history=history,
        )
        if not scope_unlocked:
            await _remember_scope_verdict(
                redis=redis,
                settings=settings,
                session_id=payload.session_id,
                scope=gates.scope,
                scope_gate=scope_gate,
            )
        for decision in _gate_attempts(gates.scope):
            metrics.record_stage(
                "scope_gate_input",
                decision.latency_ms,
                provider=decision.provider,
                model=decision.model,
                passed=decision.passed,
                success=decision.success,
                status="completed" if decision.success else "failed",
                error_type=decision.error_type,
                error_message=decision.error_message,
            )
            _log_gate_decision(request_row.id, decision, "input")
        for decision in _gate_attempts(gates.censorship):
            metrics.record_stage(
                "censorship_gate_input",
                decision.latency_ms,
                provider=decision.provider,
                model=decision.model,
                passed=decision.passed,
                success=decision.success,
                status="completed" if decision.success else "failed",
                error_type=decision.error_type,
                error_message=decision.error_message,
            )
            _log_gate_decision(request_row.id, decision, "input")
        metrics.set_count(
            "gate_calls",
            len(_gate_attempts(gates.scope)) + len(_gate_attempts(gates.censorship)),
        )
        status, rejection_reason, rejection_answer = _outcome(gates)
        report_progress(
            "routing",
            "Using the geo-agent route." if status == "completed" else "Request was rejected.",
            status="completed",
        )

        logger.info(
            "input_gates_complete request_id=%s session_id=%s scope=%s censorship=%s",
            request_row.id,
            payload.session_id,
            gates.scope.model_dump(mode="json"),
            gates.censorship.model_dump(mode="json"),
        )

        llm_response: LLMResponse | None = None
        orchestration: OrchestratorRun | None = None
        grounding: GroundingReport | None = None
        sources: list[SourceCitation] = []
        map_places: list[MapPlace] = []
        search_areas: list[ResolvedSearchArea] = []
        answer = rejection_answer

        if status == "completed":

            async def validate_answer_refs(
                candidate: str,
                executed_calls: Sequence[ExecutedToolCall],
            ) -> str | None:
                """Reject invented refs and omitted refs for concrete listed places."""

                report = await verify_answer_refs(
                    candidate,
                    place_store=place_store,
                    source_store=source_store,
                )
                if not report.grounded:
                    invented = ", ".join(report.unknown_refs)
                    return (
                        f"These references do not exist: {invented}. Rewrite the answer using "
                        "only exact [[plc_...]] and [[src_...]] refs returned by the tools in "
                        "this request. Omit a reference if no valid replacement exists."
                    )

                if _missing_required_web_citation(report, executed_calls):
                    return (
                        "The answer relies on a successful web_search but cites none of its "
                        "evidence. Rewrite the answer and append exact returned [[src_...]] refs "
                        "next to every claim supported by web_search. Keep any required "
                        "[[plc_...]] refs for concrete places as well."
                    )

                missing_names = missing_listed_place_names(
                    candidate,
                    report.place_refs,
                    (executed.result for executed in executed_calls),
                )

                if not missing_names:
                    return None
                listed = ", ".join(missing_names[:5])
                return (
                    "Your answer names concrete places returned by places_search but omits "
                    f"their required inline place refs ({listed}). Rewrite the answer and append "
                    "each included place's exact [[plc_...]] ref from the tool result. Do not add "
                    "refs for places omitted from the answer."
                )

            def report_orchestrator_progress(kind: str, details: dict[str, Any]) -> None:
                turn_index = details.get("turn_index")
                if kind == "model_started":
                    report_progress(
                        "react",
                        f"ReAct step {turn_index}: deciding the next action…",
                        step=turn_index,
                    )
                elif kind == "tool_started":
                    tool_name = str(details.get("tool_name") or "unknown")
                    report_progress(
                        "tool",
                        f"Running tool {tool_name}…",
                        step=turn_index,
                        tool_name=tool_name,
                        tool_input=details.get("tool_input"),
                    )
                elif kind == "tool_finished":
                    tool_name = str(details.get("tool_name") or "unknown")
                    ok = bool(details.get("ok"))
                    report_progress(
                        "tool",
                        f"Tool {tool_name} {'completed' if ok else 'failed'}.",
                        status="completed" if ok else "error",
                        step=turn_index,
                        tool_name=tool_name,
                        ok=ok,
                    )
                elif kind == "answer_validation":
                    message = (
                        "The model returned an empty response; retrying…"
                        if details.get("reason") == "empty_response"
                        else "Validating references and regenerating the answer…"
                    )
                    report_progress(
                        "react",
                        message,
                        step=turn_index,
                    )
                elif kind == "answer_ready":
                    report_progress(
                        "react",
                        "Model produced the final answer.",
                        status="completed",
                        step=turn_index,
                    )
                elif kind == "answer_reset" and progress_callback is not None:
                    progress_callback({"type": "answer_reset"})

            def report_answer_chunk(chunk: str) -> None:
                if progress_callback is None:
                    return
                progress_callback({"type": "answer_delta", "delta": chunk})

            # Same history the gates saw: the model needs it to resolve a
            # follow-up and to reuse refs it already obtained.
            with metrics.stage("orchestrator"):
                orchestration_runs: list[OrchestratorRun] = []
                retry_notice_visible = False

                def report_retry_answer_chunk(chunk: str) -> None:
                    nonlocal retry_notice_visible
                    if retry_notice_visible and progress_callback is not None:
                        progress_callback({"type": "answer_reset"})
                        retry_notice_visible = False
                    report_answer_chunk(chunk)

                for pipeline_attempt in range(2):
                    # A new instance means a clean ReAct message list, step
                    # budget and tool-observation state for the outer retry.
                    orchestrator = Orchestrator(
                        llm=get_llm_client(settings),
                        tool_executor=tool_executor or ToolExecutor({}),
                        tool_specs=tool_specs,
                        temperature=settings.llm_temperature,
                        top_p=settings.llm_top_p,
                        echo_store=redis,
                        answer_validator=validate_answer_refs,
                        progress_callback=report_orchestrator_progress,
                        answer_chunk_callback=(
                            report_answer_chunk
                            if pipeline_attempt == 0
                            else report_retry_answer_chunk
                        ),
                    )
                    try:
                        current_run = await orchestrator.run(model_message, history=history)
                    except OrchestratorExecutionError as exc:
                        if pipeline_attempt > 0 or not _is_llm_timeout(exc):
                            # Include the earlier attempt when the clean retry
                            # also fails, so failure observability is complete.
                            if orchestration_runs:
                                previous = _combine_orchestrator_runs(orchestration_runs)
                                turn_offset = previous.llm_turns
                                step_offset = (
                                    max(
                                        (call.step_index for call in previous.tool_calls),
                                        default=-1,
                                    )
                                    + 1
                                )
                                exc.llm_calls[:] = [
                                    *previous.llm_calls,
                                    *(
                                        replace(call, turn_index=call.turn_index + turn_offset)
                                        for call in exc.llm_calls
                                    ),
                                ]
                                exc.tool_calls[:] = [
                                    *previous.tool_calls,
                                    *(
                                        replace(call, step_index=call.step_index + step_offset)
                                        for call in exc.tool_calls
                                    ),
                                ]
                                exc.turns += previous.llm_turns
                                exc.regenerations += previous.regenerations
                            raise
                        orchestration_runs.append(_failed_orchestrator_run(exc))
                        if progress_callback is not None:
                            progress_callback({"type": "answer_reset"})
                            progress_callback(
                                {"type": "answer_delta", "delta": _EMPTY_RESPONSE_RETRY_MESSAGE}
                            )
                            retry_notice_visible = True
                        report_progress(
                            "react",
                            "The model timed out; restarting the agent pipeline…",
                            attempt=2,
                        )
                        logger.warning(
                            "llm_timeout_pipeline_retrying request_id=%s attempt=2",
                            request_row.id,
                        )
                        continue

                    orchestration_runs.append(current_run)
                    if not current_run.empty_response:
                        break
                    if pipeline_attempt == 0:
                        if progress_callback is not None:
                            progress_callback({"type": "answer_reset"})
                            progress_callback(
                                {"type": "answer_delta", "delta": _EMPTY_RESPONSE_RETRY_MESSAGE}
                            )
                            retry_notice_visible = True
                        report_progress(
                            "react",
                            "The model needs more time; restarting the agent pipeline…",
                            attempt=2,
                        )
                        logger.warning(
                            "empty_pipeline_response_retrying request_id=%s attempt=2",
                            request_row.id,
                        )
                        continue

                    # Bound the clean retry to one attempt. The waiting notice is
                    # transient UI state; a second exhausted run gets a distinct
                    # durable failure answer.
                    if retry_notice_visible and progress_callback is not None:
                        progress_callback({"type": "answer_reset"})
                        progress_callback(
                            {"type": "answer_delta", "delta": _EMPTY_RESPONSE_FAILURE_MESSAGE}
                        )
                        retry_notice_visible = False
                    fallback_trace = current_run.llm_response.trace.model_copy(
                        update={
                            "steps": [
                                *current_run.llm_response.trace.steps,
                                ReActStep(
                                    type=ReActStepType.FINAL_ANSWER,
                                    content=_EMPTY_RESPONSE_FAILURE_MESSAGE,
                                ),
                            ],
                            "final_answer": _EMPTY_RESPONSE_FAILURE_MESSAGE,
                        }
                    )
                    current_run.llm_response = current_run.llm_response.model_copy(
                        update={
                            "content": _EMPTY_RESPONSE_FAILURE_MESSAGE,
                            "trace": fallback_trace,
                        }
                    )
                    break

                orchestration = _combine_orchestrator_runs(orchestration_runs)
            llm_response = orchestration.llm_response
            answer = llm_response.content
            search_areas = reusable_search_areas(
                executed.result for executed in orchestration.tool_calls
            )

            # Echo-grounding: every ref the answer cites must resolve to a record
            # a tool actually minted. An unresolvable ref was invented.
            report_progress("grounding", "Verifying tool references…")
            with metrics.stage("grounding_verification"):
                grounding = await verify_answer_refs(
                    answer, place_store=place_store, source_store=source_store
                )
                sources = await _cited_web_sources(
                    grounding,
                    source_store,
                    orchestration.tool_calls,
                )
                selected_place_refs = selected_eligible_place_refs(
                    grounding.place_refs,
                    (executed.result for executed in orchestration.tool_calls),
                )
                search_anchor_refs = selected_search_anchor_refs(
                    selected_place_refs,
                    (executed.result for executed in orchestration.tool_calls),
                )
                map_places = await load_map_places(
                    selected_place_refs,
                    anchor_refs=search_anchor_refs,
                    place_store=place_store,
                )
            # Refs have now served their purpose: source links are returned in
            # AgentResponse.sources and rendered below the answer, not as raw
            # implementation tokens inside natural-language prose.
            answer = _hide_internal_refs(answer)
            llm_response = llm_response.model_copy(update={"content": answer})
            if not grounding.grounded:
                logger.warning(
                    "ungrounded_refs request_id=%s unknown=%s",
                    request_row.id,
                    grounding.unknown_refs,
                )

            # Post-generation check: run the SAME censorship gate on the answer,
            # so a harmful completion is caught even when the input passed.
            output_decision = await evaluate_output_censorship(
                answer, censorship_gate=censorship_gate
            )
            report_progress("grounding", "Answer checks completed.", status="completed")
            gates = gates.model_copy(update={"output_censorship": output_decision})
            for decision in _gate_attempts(output_decision):
                metrics.record_stage(
                    "censorship_gate_output",
                    decision.latency_ms,
                    provider=decision.provider,
                    model=decision.model,
                    passed=decision.passed,
                    success=decision.success,
                    status="completed" if decision.success else "failed",
                    error_type=decision.error_type,
                    error_message=decision.error_message,
                )
                _log_gate_decision(request_row.id, decision, "output")
                metrics.increment("gate_calls")
            if not output_decision.passed:
                if progress_callback is not None:
                    progress_callback({"type": "answer_reset"})
                status = "rejected"
                rejection_reason = "output_censorship"
                answer = _CENSORSHIP_ANSWER
                # Citations extracted from a blocked completion are not part of
                # the public response and must not enter restored history.
                sources = []
                map_places = []
                search_areas = []

        report_progress("persistence", "Saving conversation state…")

        raw_trace: dict[str, object] = {
            "input_gates": _decision_payload(gates),
            "pipeline_status": status,
            "rejection_reason": rejection_reason,
            "llm_trace": (
                llm_response.trace.model_dump(mode="json") if llm_response is not None else None
            ),
            "grounding": grounding.model_dump(mode="json") if grounding is not None else None,
            "selected_place_refs": [
                place.ref for place in map_places if place.marker_role == "result"
            ],
        }
        trace_row = ReactTrace(
            request_id=request_row.id,
            num_steps=len(llm_response.trace.steps) if llm_response is not None else 0,
            raw_trace=raw_trace,
        )
        db.add(trace_row)
        with metrics.stage("db_trace_flush"):
            await db.flush()
        _persist_gate_logs(
            db=db,
            request_row=request_row,
            gates=gates,
        )

        total_tokens = 0
        num_model_calls = 0
        num_tool_calls = 0
        if llm_response is not None and orchestration is not None:
            db.add(
                ModelResponse(
                    trace_id=trace_row.id,
                    step_index=len(orchestration.llm_response.trace.steps),
                    model=llm_response.model,
                    mode=llm_response.mode,
                    prompt={"user": payload.message},
                    content=llm_response.content,
                    usage=llm_response.usage.model_dump(mode="json"),
                    latency_ms=orchestration.llm_latency_ms,
                )
            )
            # One row per tool the model actually invoked in the loop.
            for llm_call in orchestration.llm_calls:
                db.add(
                    LLMCallMetric(
                        trace_id=trace_row.id,
                        turn_index=llm_call.turn_index,
                        model=llm_call.model,
                        mode=llm_call.mode,
                        latency_ms=llm_call.latency_ms,
                        prompt_tokens=llm_call.prompt_tokens,
                        completion_tokens=llm_call.completion_tokens,
                        total_tokens=llm_call.total_tokens,
                        success=llm_call.success,
                        error_type=llm_call.error_type,
                        error_message=llm_call.error_message,
                    )
                )

            for executed in orchestration.tool_calls:
                result = executed.result
                tool_metrics = result.metrics
                tool_row = ToolCall(
                    trace_id=trace_row.id,
                    step_index=executed.step_index,
                    tool_name=executed.tool_name,
                    tool_hash=result.tool_hash,
                    arguments=executed.arguments,
                    result=result.model_dump(mode="json"),
                    status="ok" if result.ok else "error",
                    error_code=result.error_code.value if result.error_code is not None else None,
                    provider=result.provider,
                    status_code=result.status_code,
                    failure_kind=(
                        result.failure_kind.value if result.failure_kind is not None else None
                    ),
                    provider_code=result.provider_code,
                    retryable=result.retryable,
                    latency_ms=tool_metrics.latency_ms if tool_metrics else None,
                    upstream_call_count=len(tool_metrics.upstream_calls) if tool_metrics else 0,
                    upstream_latency_ms=tool_metrics.upstream_latency_ms if tool_metrics else None,
                    response_bytes=tool_metrics.response_bytes if tool_metrics else 0,
                    model_tokens_estimate=(
                        tool_metrics.model_tokens_estimate if tool_metrics else 0
                    ),
                )
                if tool_metrics is not None:
                    for call_index, upstream in enumerate(tool_metrics.upstream_calls, start=1):
                        tool_row.upstream_calls.append(
                            UpstreamCallMetric(
                                call_index=call_index,
                                provider=upstream.provider,
                                operation=upstream.operation,
                                latency_ms=upstream.latency_ms,
                                parallel_group=upstream.parallel_group,
                                outcome=upstream.outcome.value,
                                status_code=upstream.status_code,
                                error_code=upstream.error_code,
                                failure_kind=upstream.failure_kind,
                                provider_code=upstream.provider_code,
                                retryable=upstream.retryable,
                            )
                        )
                db.add(tool_row)
            total_tokens = llm_response.usage.total_tokens
            num_model_calls = orchestration.llm_turns
            num_tool_calls = len(orchestration.tool_calls)
            metrics.set_count("llm_calls", num_model_calls)
            metrics.set_count("tool_calls", num_tool_calls)
            metrics.set_count(
                "successful_tool_calls",
                sum(1 for call in orchestration.tool_calls if call.result.ok),
            )
            metrics.set_count(
                "failed_tool_calls",
                sum(1 for call in orchestration.tool_calls if not call.result.ok),
            )
            metrics.set_count("upstream_calls", orchestration.upstream_call_count)
            tool_call_counts = dict(Counter(call.tool_name for call in orchestration.tool_calls))
            for tool_name, count in tool_call_counts.items():
                metrics.set_count(f"tool.{tool_name}", count)
            logger.info(
                "tool_call_summary request_id=%s total=%s by_name=%s",
                request_row.id,
                num_tool_calls,
                tool_call_counts,
            )
        else:
            tool_call_counts = {}

        gate_decisions = [decision for decision, _phase in _all_gate_attempts(gates)]
        gate_latency_ms = sum(decision.latency_ms for decision in gate_decisions)
        num_gate_calls = len(gate_decisions)
        metrics.set_count("gate_calls", num_gate_calls)

        request_row.status = status if rejection_reason is None else f"rejected_{rejection_reason}"
        with metrics.stage("db_conversation_write"):
            await durable_conversations.persist_exchange(
                session_id=payload.session_id,
                request_id=request_row.id,
                user_message=payload.message,
                assistant_message=answer,
                sources=sources,
                map_places=map_places,
                search_areas=search_areas,
                status=status,
                rejection_reason=rejection_reason,
            )
        total_latency_ms = int((perf_counter() - started) * 1000)
        db.add(
            Metrics(
                request_id=request_row.id,
                total_latency_ms=total_latency_ms,
                total_tokens=total_tokens,
                num_tool_calls=num_tool_calls,
                num_model_calls=num_model_calls,
                success=True,
                llm_latency_ms=(orchestration.llm_latency_ms if orchestration else 0),
                tool_latency_ms=(orchestration.tool_latency_ms if orchestration else 0),
                upstream_call_count=(orchestration.upstream_call_count if orchestration else 0),
                successful_tool_calls=metrics.counts.get("successful_tool_calls", 0),
                failed_tool_calls=metrics.counts.get("failed_tool_calls", 0),
                regenerations=(orchestration.regenerations if orchestration else 0),
                gate_latency_ms=gate_latency_ms,
                num_gate_calls=num_gate_calls,
                extra={
                    "pipeline_status": status,
                    "rejection_reason": rejection_reason,
                    "gate_provider": {
                        "scope": gates.scope.provider,
                        "censorship": gates.censorship.provider,
                    },
                    "gate_latency_ms": {
                        "scope_input": gates.scope.latency_ms,
                        "censorship_input": gates.censorship.latency_ms,
                        "censorship_output": (
                            gates.output_censorship.latency_ms
                            if gates.output_censorship is not None
                            else None
                        ),
                    },
                    "tool_call_counts": tool_call_counts,
                    "grounding": (
                        {
                            "refs": grounding.total_refs,
                            "grounded": grounding.grounded,
                            "unknown_refs": grounding.unknown_refs,
                        }
                        if grounding is not None
                        else None
                    ),
                    "map_places": len(map_places),
                    "regenerations": (
                        orchestration.regenerations if orchestration is not None else 0
                    ),
                    "observability": metrics.snapshot(),
                    "llm_call_latency_ms": (
                        orchestration.llm_call_latencies_ms if orchestration is not None else []
                    ),
                    "llm_calls": (
                        [
                            {
                                "turn_index": call.turn_index,
                                "completion_tokens": call.completion_tokens,
                                "reasoning_tokens": call.reasoning_tokens,
                                "visible_completion_tokens": max(
                                    0, call.completion_tokens - call.reasoning_tokens
                                ),
                                "finish_reason": call.finish_reason,
                            }
                            for call in orchestration.llm_calls
                        ]
                        if orchestration is not None
                        else []
                    ),
                    "llm_latency_ms": (
                        orchestration.llm_latency_ms if orchestration is not None else 0
                    ),
                    "tool_latency_ms": (
                        orchestration.tool_latency_ms if orchestration is not None else 0
                    ),
                    "upstream_call_count": (
                        orchestration.upstream_call_count if orchestration is not None else 0
                    ),
                },
            )
        )
        # Persist every measured atomic stage. DB commit itself is intentionally
        # not logged as a stage.
        for measurement in metrics.measurements:
            db.add(
                PipelineStageMetric(
                    request_id=request_row.id,
                    stage_name=measurement.name,
                    occurrence=measurement.occurrence,
                    latency_ms=measurement.latency_ms,
                    status=measurement.status,
                    details=measurement.fields or None,
                    error_type=measurement.error_type,
                    error_message=measurement.error_message,
                )
            )

        await db.commit()

        # Cache state is compatibility/acceleration only. It is intentionally
        # written after the durable transaction so a DB rollback cannot leave a
        # phantom conversation in Redis. Cache loss must not turn a committed
        # answer into an API failure either.
        try:
            await _cache_pipeline_state(
                redis=redis,
                payload=payload,
                gates=gates,
                status=status,
                answer=answer,
            )
        except Exception:
            logger.warning(
                "conversation_cache_write_failed request_id=%s session_id=%s",
                request_row.id,
                payload.session_id,
                exc_info=True,
            )
        report_progress("persistence", "Conversation state saved.", status="completed")

        logger.info(
            "pipeline_complete request_id=%s status=%s rejection_reason=%s "
            "main_llm_called=%s latency_ms=%s gate_latency_ms=%s stage_latency_ms=%s"
            "counts=%s tool_call_counts=%s",
            request_row.id,
            status,
            rejection_reason,
            llm_response is not None,
            total_latency_ms,
            gate_latency_ms,
            metrics.stages_ms,
            metrics.counts,
            tool_call_counts,
        )
        return AgentResponse(
            request_id=str(request_row.id),
            session_id=payload.session_id,
            status=status,
            rejection_reason=rejection_reason,
            answer=answer,
            gates=gates,
            llm=llm_response,
            sources=sources,
            map=MapData(places=map_places) if map_places else None,
        )
    except Exception as exc:
        request_id = request_row.id
        logger.exception(
            "pipeline_failed request_id=%s session_id=%s",
            request_id,
            payload.session_id,
        )
        await db.rollback()

        # The failed transaction is gone, so persist observability in a fresh
        # transaction. This keeps diagnostics for backend errors that are caught
        # and converted to an HTTP error by the API layer.
        try:
            failed_request = Request(
                id=request_id,
                session_id=payload.session_id,
                user_query=payload.message,
                status="failed",
            )
            db.add(failed_request)
            await db.flush()

            for measurement in metrics.measurements:
                db.add(
                    PipelineStageMetric(
                        request_id=request_id,
                        stage_name=measurement.name,
                        occurrence=measurement.occurrence,
                        latency_ms=measurement.latency_ms,
                        status=measurement.status,
                        details=measurement.fields or None,
                        error_type=measurement.error_type,
                        error_message=measurement.error_message,
                    )
                )

            failed_trace = ReactTrace(
                request_id=request_id,
                num_steps=0,
                raw_trace={
                    "pipeline_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            db.add(failed_trace)
            await db.flush()

            partial_llm_calls = []
            partial_tool_calls = []
            partial_regenerations = 0
            if isinstance(exc, OrchestratorExecutionError):
                partial_llm_calls = exc.llm_calls
                partial_tool_calls = exc.tool_calls
                partial_regenerations = exc.regenerations
            elif orchestration is not None:
                partial_llm_calls = orchestration.llm_calls
                partial_tool_calls = orchestration.tool_calls
                partial_regenerations = orchestration.regenerations

            for call in partial_llm_calls:
                db.add(
                    LLMCallMetric(
                        trace_id=failed_trace.id,
                        turn_index=call.turn_index,
                        model=call.model,
                        mode=call.mode,
                        latency_ms=call.latency_ms,
                        prompt_tokens=call.prompt_tokens,
                        completion_tokens=call.completion_tokens,
                        total_tokens=call.total_tokens,
                        success=call.success,
                        error_type=call.error_type,
                        error_message=call.error_message,
                    )
                )

            for executed in partial_tool_calls:
                result = executed.result
                tool_metrics = result.metrics
                tool_row = ToolCall(
                    trace_id=failed_trace.id,
                    step_index=executed.step_index,
                    tool_name=executed.tool_name,
                    tool_hash=result.tool_hash,
                    arguments=executed.arguments,
                    result=result.model_dump(mode="json"),
                    status="ok" if result.ok else "error",
                    error_code=result.error_code.value if result.error_code is not None else None,
                    provider=result.provider,
                    status_code=result.status_code,
                    failure_kind=(
                        result.failure_kind.value if result.failure_kind is not None else None
                    ),
                    provider_code=result.provider_code,
                    retryable=result.retryable,
                    latency_ms=tool_metrics.latency_ms if tool_metrics else None,
                    upstream_call_count=len(tool_metrics.upstream_calls) if tool_metrics else 0,
                    upstream_latency_ms=tool_metrics.upstream_latency_ms if tool_metrics else None,
                    response_bytes=tool_metrics.response_bytes if tool_metrics else 0,
                    model_tokens_estimate=tool_metrics.model_tokens_estimate if tool_metrics else 0,
                )
                if tool_metrics is not None:
                    for call_index, upstream in enumerate(tool_metrics.upstream_calls, start=1):
                        tool_row.upstream_calls.append(
                            UpstreamCallMetric(
                                call_index=call_index,
                                provider=upstream.provider,
                                operation=upstream.operation,
                                latency_ms=upstream.latency_ms,
                                parallel_group=upstream.parallel_group,
                                outcome=upstream.outcome.value,
                                status_code=upstream.status_code,
                                error_code=upstream.error_code,
                                failure_kind=upstream.failure_kind,
                                provider_code=upstream.provider_code,
                                retryable=upstream.retryable,
                            )
                        )
                db.add(tool_row)

            if gates is not None:
                _persist_gate_logs(db=db, request_row=failed_request, gates=gates)
                gate_decisions = [decision for decision, _phase in _all_gate_attempts(gates)]
            else:
                gate_decisions = []

            db.add(
                Metrics(
                    request_id=request_id,
                    total_latency_ms=int((perf_counter() - started) * 1000),
                    total_tokens=sum(call.total_tokens for call in partial_llm_calls),
                    num_tool_calls=len(partial_tool_calls),
                    num_model_calls=len(partial_llm_calls),
                    success=False,
                    llm_latency_ms=sum(call.latency_ms for call in partial_llm_calls),
                    tool_latency_ms=sum(
                        call.result.metrics.latency_ms
                        for call in partial_tool_calls
                        if call.result.metrics is not None
                    ),
                    upstream_call_count=sum(
                        len(call.result.metrics.upstream_calls)
                        for call in partial_tool_calls
                        if call.result.metrics is not None
                    ),
                    successful_tool_calls=sum(1 for call in partial_tool_calls if call.result.ok),
                    failed_tool_calls=sum(1 for call in partial_tool_calls if not call.result.ok),
                    regenerations=partial_regenerations,
                    gate_latency_ms=sum(decision.latency_ms for decision in gate_decisions),
                    num_gate_calls=len(gate_decisions),
                    extra={
                        "pipeline_status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "observability": metrics.snapshot(),
                    },
                )
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("failed_observability_persistence request_id=%s", request_id)
        raise
