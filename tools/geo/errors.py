"""Provider-independent geo domain errors shared across subpackages."""

from tools.base import ToolClarification, ToolErrorCode, ToolExecutionError


class AmbiguousPlaceError(ToolExecutionError):
    """More than one candidate plausibly matches a textual place."""

    def __init__(
        self,
        query: str,
        *,
        clarification: ToolClarification | None = None,
    ) -> None:
        super().__init__(
            ToolErrorCode.INVALID_INPUT,
            f"Place is ambiguous: {query!r}. Specify a more precise name or full address.",
            clarification=clarification,
        )
