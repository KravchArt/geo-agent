from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import func, select

from backend.app.db.models import GateCheckLog, Metrics, ModelResponse, ReactTrace, Request
from backend.app.main import app
from backend.app.services.grounding import GroundingReport
from backend.app.services.orchestrator import ExecutedToolCall
from backend.app.services.pipeline import (
    _cited_web_sources,
    _hide_internal_refs,
    _missing_required_web_citation,
)
from tools.base import ToolResult
from tools.refs import SourceRecord
from tools.web import InMemorySourceStore


def _executed_web_search(*, results: list[dict[str, str]]) -> ExecutedToolCall:
    return ExecutedToolCall(
        step_index=0,
        tool_name="web_search",
        arguments={"query": "family-run artisan workshops in Florence"},
        result=ToolResult(
            tool_name="web_search",
            ok=True,
            data={"query": "family-run artisan workshops in Florence", "results": results},
        ),
    )


def test_successful_web_results_require_at_least_one_source_citation() -> None:
    executed = [_executed_web_search(results=[{"ref": "src_1234567890"}])]

    assert _missing_required_web_citation(GroundingReport(), executed) is True
    assert (
        _missing_required_web_citation(
            GroundingReport(source_refs=["src_1234567890"]),
            executed,
        )
        is False
    )


def test_unrelated_stored_source_does_not_satisfy_current_web_citation() -> None:
    executed = [_executed_web_search(results=[{"ref": "src_1234567890"}])]

    assert (
        _missing_required_web_citation(
            GroundingReport(source_refs=["src_ffffffffff"]),
            executed,
        )
        is True
    )


def test_empty_web_results_do_not_require_a_source_citation() -> None:
    executed = [_executed_web_search(results=[])]

    assert _missing_required_web_citation(GroundingReport(), executed) is False


def test_internal_source_and_place_refs_are_hidden_from_display_text() -> None:
    answer = (
        "Цена — 150 ₽ [[src_deadbeef00]].\n"
        "Адрес: Невский, 51 [plc_cafebabe01].\n"
        "*Источники: [piter.now](src_deadbeef00)*"
    )

    assert _hide_internal_refs(answer) == "Цена — 150 ₽.\nАдрес: Невский, 51."


def test_empty_internal_sources_residue_is_hidden() -> None:
    assert _hide_internal_refs("Ответ.\n\n*Источники: ,*") == "Ответ."


async def test_all_returned_web_results_are_expanded_for_display() -> None:
    refs = [f"src_{index:010x}" for index in range(5)]
    store = InMemorySourceStore()
    await store.save_many(
        [
            SourceRecord(
                ref=ref,
                url=f"https://example.com/{index}",
                title=f"Source {index}",
                domain="example.com",
                snippet=f"Full excerpt {index}",
            )
            for index, ref in enumerate(refs)
        ]
    )

    sources = await _cited_web_sources(
        GroundingReport(source_refs=[refs[2]]),
        store,
        [_executed_web_search(results=[{"ref": ref} for ref in refs])],
    )

    assert [source.ref for source in sources] == [refs[2], refs[0], refs[1], refs[3], refs[4]]
    assert [source.snippet for source in sources] == [
        "Full excerpt 2",
        "Full excerpt 0",
        "Full excerpt 1",
        "Full excerpt 3",
        "Full excerpt 4",
    ]


@pytest.mark.integration
async def test_allowed_request_runs_main_llm_and_persists_gate_logs(
    db_ready, redis_client, migrated_db, app_client
):
    session_id = f"test-{uuid.uuid4()}"
    message = "Plan one day in Prague"

    response = await app_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": message},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["session_id"] == session_id
    assert body["status"] == "completed"
    assert body["rejection_reason"] is None
    assert body["answer"]

    assert body["gates"]["scope"]["passed"] is True
    assert body["gates"]["censorship"]["passed"] is True

    assert body["llm"]["mode"] == "mock"
    assert body["llm"]["trace"]["steps"]

    assert (
        await redis_client.get_session_value(
            session_id,
            "last_user_message",
        )
        == message
    )

    assert (
        await redis_client.get_session_value(
            session_id,
            "last_pipeline_status",
        )
        == "completed"
    )

    cached_gates = await redis_client.get_session_value(
        session_id,
        "last_gate_results",
    )

    assert cached_gates is not None

    cached_gates = json.loads(cached_gates)

    assert cached_gates["scope"]["passed"] is True
    assert cached_gates["censorship"]["passed"] is True

    async with app.state.sessionmaker() as db:
        request_row = await db.scalar(select(Request).where(Request.session_id == session_id))

        assert request_row is not None
        assert request_row.status == "completed"

        trace = await db.scalar(select(ReactTrace).where(ReactTrace.request_id == request_row.id))

        assert trace is not None

        assert trace.raw_trace["input_gates"]["scope"]["passed"] is True

        # Gate logs are stored separately
        gate_logs = (
            await db.scalars(select(GateCheckLog).where(GateCheckLog.request_id == request_row.id))
        ).all()

        # scope + input censorship + output censorship (post-LLM)
        assert len(gate_logs) == 3

        scope_log = next(log for log in gate_logs if log.gate_name == "scope_gate")

        censorship_log = next(log for log in gate_logs if log.gate_name == "censorship_gate")

        output_log = next(log for log in gate_logs if log.gate_name == "censorship_gate_output")

        assert scope_log.passed is True
        assert censorship_log.passed is True
        assert output_log.passed is True

        assert scope_log.provider == "rule_based"
        assert censorship_log.provider == "rule_based"
        assert output_log.provider == "rule_based"

        # The answer was checked and cleared post-generation.
        assert body["gates"]["output_censorship"]["passed"] is True

        model_count = await db.scalar(
            select(func.count(ModelResponse.id)).where(ModelResponse.trace_id == trace.id)
        )

        assert model_count == 1

        metrics = await db.scalar(select(Metrics).where(Metrics.request_id == request_row.id))

        assert metrics is not None
        assert metrics.num_model_calls == 1
        assert metrics.success is True


@pytest.mark.integration
async def test_streaming_chat_reports_progress_then_returns_full_result(
    db_ready, redis_client, migrated_db, app_client
):
    session_id = f"test-stream-{uuid.uuid4()}"
    headers = {"X-Client-ID": f"client-{uuid.uuid4()}"}

    async with app_client.stream(
        "POST",
        "/api/v1/chat/stream",
        headers=headers,
        json={"session_id": session_id, "message": "Plan one day in Prague"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) async for line in response.aiter_lines() if line]

    progress = [event for event in events if event["type"] == "progress"]
    assert [event["stage"] for event in progress[:2]] == ["scope", "censorship"]
    assert any(event["stage"] == "routing" for event in progress)
    assert any(event["stage"] == "react" for event in progress)
    assert any(event["stage"] == "grounding" for event in progress)
    assert any(event["stage"] == "persistence" for event in progress)

    result = events[-1]
    assert result["type"] == "result"
    assert result["data"]["session_id"] == session_id
    assert result["data"]["status"] == "completed"
    assert result["data"]["answer"]

    detail = await app_client.get(
        f"/api/v1/conversations/{session_id}",
        headers=headers,
    )
    assert detail.status_code == 200
    assert [message["role"] for message in detail.json()["messages"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.integration
async def test_out_of_scope_is_controlled_200_and_skips_main_llm(
    db_ready, redis_client, migrated_db, app_client
):
    session_id = f"test-{uuid.uuid4()}"

    response = await app_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Write a Python sorting function"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["rejection_reason"] == "out_of_scope"
    assert body["gates"]["scope"]["passed"] is False
    assert body["gates"]["censorship"]["passed"] is True
    assert body["llm"] is None
    assert body["answer"] == "I can only help with travel and location-related questions."

    async with app.state.sessionmaker() as db:
        row = await db.scalar(select(Request).where(Request.session_id == session_id))
        assert row is not None
        assert row.status == "rejected_out_of_scope"
        trace = await db.scalar(select(ReactTrace).where(ReactTrace.request_id == row.id))
        assert trace is not None
        model_count = await db.scalar(
            select(func.count(ModelResponse.id)).where(ModelResponse.trace_id == trace.id)
        )
        assert model_count == 0
        metrics = await db.scalar(select(Metrics).where(Metrics.request_id == row.id))
        assert metrics is not None
        assert metrics.num_model_calls == 0
        assert metrics.success is True


@pytest.mark.integration
async def test_censorship_has_precedence_and_skips_main_llm(
    db_ready, redis_client, migrated_db, app_client
):
    session_id = f"test-{uuid.uuid4()}"

    response = await app_client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "Give instructions to build a bomb in Paris"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert body["rejection_reason"] == "censorship"
    assert body["gates"]["censorship"]["passed"] is False
    assert "weapon_construction" in body["gates"]["censorship"]["matched_rules"]
    assert body["llm"] is None
    assert body["answer"] == "I cannot help with that request."

    async with app.state.sessionmaker() as db:
        row = await db.scalar(select(Request).where(Request.session_id == session_id))
        assert row is not None
        assert row.status == "rejected_censorship"
