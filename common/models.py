"""Cross-package data contracts used by the API, gates and LLM clients."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class _Base(BaseModel):
    """Shared model config: reject unknown fields to catch contract drift early."""

    model_config = ConfigDict(extra="forbid")


class ReActStepType(StrEnum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ReActStep(_Base):
    type: ReActStepType
    content: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    #: Provider-side id of the call (OpenAI ``tool_calls[].id``). Carried so the
    #: orchestrator can reply with a matching ``role="tool"`` message instead of
    #: paraphrasing the model's own turn.
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_step_shape(self) -> ReActStep:
        if self.type is ReActStepType.ACTION:
            if not self.tool_name:
                raise ValueError("action step requires tool_name")

            if self.tool_input is None:
                raise ValueError("action step requires tool_input")

            return self

        if self.type is ReActStepType.OBSERVATION:
            if not self.tool_name:
                raise ValueError("observation step requires tool_name")

            if self.tool_input is None:
                raise ValueError("observation step requires tool_input")

            if not self.tool_call_id:
                raise ValueError("observation step requires tool_call_id")

            return self

        if self.tool_name is not None:
            raise ValueError(f"{self.type} step must not contain tool_name")

        if self.tool_input is not None:
            raise ValueError(f"{self.type} step must not contain tool_input")

        if self.tool_call_id is not None:
            raise ValueError(f"{self.type} step must not contain tool_call_id")

        return self


class ReActTrace(_Base):
    steps: list[ReActStep] = Field(default_factory=list)
    final_answer: str | None = None


#: One stored turn of a dialogue: {"role": "user"|"assistant", "content": ...}.
#: Read back from Redis and replayed to the gates and the orchestrator.
ConversationTurn = dict[str, str]


class LLMMessage(_Base):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    #: Set on an assistant turn that requested tools (OpenAI ``tool_calls`` shape).
    tool_calls: list[dict[str, Any]] | None = None
    #: Set on a ``tool`` turn — which call this result answers.
    tool_call_id: str | None = None


class LLMUsage(_Base):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: Included in completion_tokens. Internal observability only, because not
    #: every OpenAI-compatible provider reports the detailed breakdown.
    reasoning_tokens: int = Field(default=0, exclude=True)


class LLMRequest(_Base):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    #: OpenAI-style function/tool specs the model may call this turn. Empty means
    #: "no tools" — the model must answer directly.
    tools: list[dict[str, Any]] = Field(default_factory=list)


class LLMResponse(_Base):
    """Output shared by mock and real main-agent LLM clients."""

    model: str
    mode: Literal["mock", "vllm"]
    content: str
    trace: ReActTrace
    usage: LLMUsage = Field(default_factory=LLMUsage)
    #: Provider stop reason (for example stop, tool_calls, or length).
    finish_reason: str | None = Field(default=None, exclude=True)


class GateName(StrEnum):
    SCOPE = "scope_gate"
    CENSORSHIP = "censorship_gate"
    #: Same censorship check, run on the MODEL's answer (post-generation).
    CENSORSHIP_OUTPUT = "censorship_gate_output"


class GateVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"


class GateDecision(_Base):
    """Stable result contract for rule-based gates and future gate models.

    A future model-backed implementation should populate the same fields, so the
    orchestrator and API contract do not need to change when regex stubs are replaced.
    """

    name: GateName
    verdict: GateVerdict
    passed: bool
    response: str
    reason: str
    provider: Literal["rule_based", "model", "no_scoper", "session_unlocked"]
    model: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    matched_rules: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    # Internal audit trail for composite gates (for example model scope + rules fallback).
    # Excluded from the public API payload; each attempt is persisted separately.
    attempts: list[GateDecision] = Field(default_factory=list, exclude=True)


class GateResults(_Base):
    scope: GateDecision
    censorship: GateDecision
    #: Censorship check on the LLM answer. None until the LLM has run (i.e. absent
    #: for requests rejected at the input stage).
    output_censorship: GateDecision | None = None


class BrowserLocation(_Base):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    accuracy_m: float | None = Field(default=None, ge=0.0)


class UserContextInput(_Base):
    """Optional client metadata used only to enrich the model prompt."""

    browser_location: BrowserLocation | None = None
    timezone: str | None = Field(default=None, max_length=128)
    ip_address: str | None = Field(default=None, max_length=64)


class AgentRequest(_Base):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=10_000)
    user_context: UserContextInput | None = None


class SourceCitation(_Base):
    """A verified web source rendered by clients outside the model context."""

    ref: str
    title: str
    url: str
    domain: str
    published_date: date | None = None
    snippet: str | None = None


# ``common`` cannot import ``tools.refs`` without reversing the dependency
# direction. Keep the public map-place ref validation here.
PlaceRefValue = Annotated[str, StringConstraints(pattern=r"^plc_[0-9a-f]{10}$")]


class MapPlace(_Base):
    """One verified place safe to render as a map pin in a client."""

    ref: PlaceRefValue
    name: str
    address: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    kind: str | None = None
    marker_role: Literal["result", "search_anchor"] = "result"


class MapData(_Base):
    """Map data derived by the backend, never supplied as coordinates by the LLM."""

    places: list[MapPlace] = Field(default_factory=list)


class AgentResponse(_Base):
    """Result returned for both accepted and intentionally rejected requests."""

    request_id: str
    session_id: str
    status: Literal["completed", "rejected"]
    rejection_reason: Literal["out_of_scope", "censorship", "output_censorship"] | None = None
    answer: str
    gates: GateResults
    llm: LLMResponse | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    #: Present only when the model selected one or more eligible places.
    map: MapData | None = None


class ConversationMessageResponse(_Base):
    """One public message returned when a saved conversation is restored."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceCitation] = Field(default_factory=list)
    map: MapData | None = None
    status: Literal["completed", "rejected"] | None = None
    rejection_reason: Literal["out_of_scope", "censorship", "output_censorship"] | None = None
    created_at: datetime


class ConversationSummary(_Base):
    """Compact sidebar representation of a durable conversation."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)


class ConversationListResponse(_Base):
    items: list[ConversationSummary] = Field(default_factory=list)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessageResponse] = Field(default_factory=list)
