import pytest
from pydantic import ValidationError

from common.models import ReActStep, ReActStepType


def test_action_step_accepts_tool_call() -> None:
    step = ReActStep(
        type=ReActStepType.ACTION,
        content="Search for places",
        tool_name="places_search",
        tool_input={"query": "old town"},
    )

    assert step.tool_name == "places_search"
    assert step.tool_input == {"query": "old town"}
    assert step.tool_call_id is None


def test_action_step_accepts_provider_tool_call_id() -> None:
    step = ReActStep(
        type=ReActStepType.ACTION,
        content="Search for places",
        tool_name="places_search",
        tool_input={"query": "old town"},
        tool_call_id="call_123",
    )

    assert step.tool_call_id == "call_123"


def test_action_step_rejects_missing_tool_name() -> None:
    with pytest.raises(ValidationError, match="action step requires tool_name"):
        ReActStep(
            type=ReActStepType.ACTION,
            content="Search for places",
            tool_input={"query": "old town"},
        )


def test_action_step_rejects_empty_tool_name() -> None:
    with pytest.raises(ValidationError, match="action step requires tool_name"):
        ReActStep(
            type=ReActStepType.ACTION,
            content="Search for places",
            tool_name="",
            tool_input={"query": "old town"},
        )


def test_action_step_rejects_missing_tool_input() -> None:
    with pytest.raises(ValidationError, match="action step requires tool_input"):
        ReActStep(
            type=ReActStepType.ACTION,
            content="Search for places",
            tool_name="places_search",
        )


def test_observation_step_accepts_tool_metadata() -> None:
    step = ReActStep(
        type=ReActStepType.OBSERVATION,
        content="Found results",
        tool_name="places_search",
        tool_input={"query": "coffee"},
        tool_call_id="call_123",
    )

    assert step.tool_name == "places_search"
    assert step.tool_input == {"query": "coffee"}
    assert step.tool_call_id == "call_123"


def test_observation_step_rejects_missing_tool_name() -> None:
    with pytest.raises(ValidationError, match="observation step requires tool_name"):
        ReActStep(
            type=ReActStepType.OBSERVATION,
            content="Found results",
            tool_input={"query": "coffee"},
            tool_call_id="call_123",
        )


def test_observation_step_rejects_empty_tool_name() -> None:
    with pytest.raises(ValidationError, match="observation step requires tool_name"):
        ReActStep(
            type=ReActStepType.OBSERVATION,
            content="Found results",
            tool_name="",
            tool_input={"query": "coffee"},
            tool_call_id="call_123",
        )


def test_observation_step_rejects_missing_tool_input() -> None:
    with pytest.raises(ValidationError, match="observation step requires tool_input"):
        ReActStep(
            type=ReActStepType.OBSERVATION,
            content="Found results",
            tool_name="places_search",
            tool_call_id="call_123",
        )


def test_observation_step_rejects_missing_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="observation step requires tool_call_id"):
        ReActStep(
            type=ReActStepType.OBSERVATION,
            content="Found results",
            tool_name="places_search",
            tool_input={"query": "coffee"},
        )


def test_observation_step_rejects_empty_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="observation step requires tool_call_id"):
        ReActStep(
            type=ReActStepType.OBSERVATION,
            content="Found results",
            tool_name="places_search",
            tool_input={"query": "coffee"},
            tool_call_id="",
        )


@pytest.mark.parametrize(
    "step_type",
    [
        ReActStepType.THOUGHT,
        ReActStepType.FINAL_ANSWER,
    ],
)
def test_plain_step_accepts_content_only(step_type: ReActStepType) -> None:
    step = ReActStep(type=step_type, content="Step content")

    assert step.tool_name is None
    assert step.tool_input is None
    assert step.tool_call_id is None


@pytest.mark.parametrize(
    "step_type",
    [
        ReActStepType.THOUGHT,
        ReActStepType.FINAL_ANSWER,
    ],
)
def test_plain_step_rejects_tool_name(step_type: ReActStepType) -> None:
    with pytest.raises(ValidationError, match="must not contain tool_name"):
        ReActStep(
            type=step_type,
            content="Consider options",
            tool_name="places_search",
        )


@pytest.mark.parametrize(
    "step_type",
    [
        ReActStepType.THOUGHT,
        ReActStepType.FINAL_ANSWER,
    ],
)
def test_plain_step_rejects_tool_input(step_type: ReActStepType) -> None:
    with pytest.raises(ValidationError, match="must not contain tool_input"):
        ReActStep(
            type=step_type,
            content="Consider options",
            tool_input={},
        )


@pytest.mark.parametrize(
    "step_type",
    [
        ReActStepType.THOUGHT,
        ReActStepType.FINAL_ANSWER,
    ],
)
def test_plain_step_rejects_tool_call_id(step_type: ReActStepType) -> None:
    with pytest.raises(ValidationError, match="must not contain tool_call_id"):
        ReActStep(
            type=step_type,
            content="Consider options",
            tool_call_id="call_123",
        )
