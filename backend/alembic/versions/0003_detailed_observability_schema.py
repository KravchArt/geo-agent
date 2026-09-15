"""normalize detailed observability metrics

Revision ID: 0003_detailed_observability
Revises: 0002_gate_check_log
Create Date: 2026-07-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_detailed_observability"
down_revision: str | None = "0002_gate_check_log"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:

    # Detailed per-gate fields (merged from the former revision 0004).
    op.add_column("gate_check_log", sa.Column("phase", sa.String(length=32), nullable=False, server_default="input"))
    op.add_column("gate_check_log", sa.Column("call_index", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("gate_check_log", sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("gate_check_log", sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("gate_check_log", sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("gate_check_log", sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("gate_check_log", sa.Column("error_type", sa.String(length=256), nullable=True))
    op.add_column("gate_check_log", sa.Column("error_message", sa.Text(), nullable=True))
    op.create_index("ix_gate_check_log_phase", "gate_check_log", ["phase"])
    op.create_unique_constraint(
        "uq_gate_check_log_request_gate_phase_call",
        "gate_check_log",
        ["request_id", "gate_name", "phase", "call_index"],
    )

    op.add_column("metrics", sa.Column("gate_latency_ms", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("num_gate_calls", sa.Integer(), nullable=False, server_default="0"))
    # Fast request-level aggregates: no JSON extraction is needed for dashboards.
    op.add_column("metrics", sa.Column("llm_latency_ms", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("tool_latency_ms", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("upstream_call_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("successful_tool_calls", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("failed_tool_calls", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("metrics", sa.Column("regenerations", sa.Integer(), nullable=False, server_default="0"))

    # Tool-level aggregates, while retaining the full serialized result for replay/debugging.
    op.add_column("tool_call", sa.Column("upstream_call_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("tool_call", sa.Column("upstream_latency_ms", sa.Integer(), nullable=True))
    op.add_column("tool_call", sa.Column("response_bytes", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("tool_call", sa.Column("model_tokens_estimate", sa.Integer(), nullable=False, server_default="0"))

    # Defaults above are only for backfilling existing rows. New writes must provide
    # explicit observability values so missing application data is never hidden.
    for table_name, column_name in [
        ("gate_check_log", "phase"),
        ("gate_check_log", "call_index"),
        ("gate_check_log", "success"),
        ("gate_check_log", "prompt_tokens"),
        ("gate_check_log", "completion_tokens"),
        ("gate_check_log", "total_tokens"),
        ("metrics", "gate_latency_ms"),
        ("metrics", "num_gate_calls"),
        ("metrics", "llm_latency_ms"),
        ("metrics", "tool_latency_ms"),
        ("metrics", "upstream_call_count"),
        ("metrics", "successful_tool_calls"),
        ("metrics", "failed_tool_calls"),
        ("metrics", "regenerations"),
        ("tool_call", "upstream_call_count"),
        ("tool_call", "response_bytes"),
        ("tool_call", "model_tokens_estimate"),
    ]:
        op.alter_column(table_name, column_name, server_default=None)

    op.create_table(
        "pipeline_stage_metric",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage_name", sa.String(length=128), nullable=False),
        sa.Column("occurrence", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("error_type", sa.String(length=256), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["request_id"], ["request.id"], name="fk_pipeline_stage_metric_request_id_request", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_pipeline_stage_metric"),
        sa.UniqueConstraint("request_id", "stage_name", "occurrence", name="uq_pipeline_stage_metric_request_stage_occurrence"),
    )
    op.create_index("ix_pipeline_stage_metric_request_id", "pipeline_stage_metric", ["request_id"])
    op.create_index("ix_pipeline_stage_metric_stage_name", "pipeline_stage_metric", ["stage_name"])
    op.create_index("ix_pipeline_stage_metric_status", "pipeline_stage_metric", ["status"])

    op.create_table(
        "llm_call_metric",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=256), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["trace_id"], ["react_trace.id"], name="fk_llm_call_metric_trace_id_react_trace", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_llm_call_metric"),
        sa.UniqueConstraint("trace_id", "turn_index", name="uq_llm_call_metric_trace_turn"),
    )
    op.create_index("ix_llm_call_metric_trace_id", "llm_call_metric", ["trace_id"])

    op.create_table(
        "upstream_call_metric",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tool_call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("call_index", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("operation", sa.String(length=128), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("parallel_group", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("failure_kind", sa.String(length=128), nullable=True),
        sa.Column("provider_code", sa.String(length=128), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["tool_call_id"], ["tool_call.id"], name="fk_upstream_call_metric_tool_call_id_tool_call", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_upstream_call_metric"),
        sa.UniqueConstraint("tool_call_id", "call_index", name="uq_upstream_call_metric_tool_call_index"),
    )
    op.create_index("ix_upstream_call_metric_tool_call_id", "upstream_call_metric", ["tool_call_id"])
    op.create_index("ix_upstream_call_metric_provider", "upstream_call_metric", ["provider"])
    op.create_index("ix_upstream_call_metric_operation", "upstream_call_metric", ["operation"])
    op.create_index("ix_upstream_call_metric_outcome", "upstream_call_metric", ["outcome"])


def downgrade() -> None:
    op.drop_column("metrics", "num_gate_calls")
    op.drop_column("metrics", "gate_latency_ms")
    op.drop_constraint("uq_gate_check_log_request_gate_phase_call", "gate_check_log", type_="unique")
    op.drop_index("ix_gate_check_log_phase", table_name="gate_check_log")
    for column in [
        "error_message", "error_type", "total_tokens", "completion_tokens",
        "prompt_tokens", "success", "call_index", "phase",
    ]:
        op.drop_column("gate_check_log", column)

    op.drop_table("upstream_call_metric")
    op.drop_table("llm_call_metric")
    op.drop_table("pipeline_stage_metric")

    op.drop_column("tool_call", "model_tokens_estimate")
    op.drop_column("tool_call", "response_bytes")
    op.drop_column("tool_call", "upstream_latency_ms")
    op.drop_column("tool_call", "upstream_call_count")

    op.drop_column("metrics", "regenerations")
    op.drop_column("metrics", "failed_tool_calls")
    op.drop_column("metrics", "successful_tool_calls")
    op.drop_column("metrics", "upstream_call_count")
    op.drop_column("metrics", "tool_latency_ms")
    op.drop_column("metrics", "llm_latency_ms")
