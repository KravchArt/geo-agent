"""Echo-grounding: prove the answer's refs came from real tool results.

The refs contract (:mod:`tools.refs`) says only a tool can mint a ``plc_``/``src_``
ref. That is a guarantee only if somebody checks it — otherwise the model can
write ``plc_deadbeef00`` and nothing notices, which is exactly the hallucination
the whole ref design exists to prevent.

This module closes the loop: pull every ref out of the finished answer and look
each one up in the stores. A ref the stores never heard of was invented.

Note what this does NOT claim: it verifies that cited refs are *real*, not that
the prose around them is faithful. It is a cheap, decisive check on the one thing
that is machine-checkable.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from tools.geo.place_store import PlaceStore
from tools.refs import PLACE_REF_PATTERN, SOURCE_REF_PATTERN
from tools.web.source_store import SourceStore

#: The ref patterns, unanchored, so they can be found inside prose.
_PLACE_RE = re.compile(PLACE_REF_PATTERN.strip("^$"))
_SOURCE_RE = re.compile(SOURCE_REF_PATTERN.strip("^$"))


class GroundingReport(BaseModel):
    """Verdict on the refs cited by one answer."""

    #: Refs found in the answer text.
    place_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    #: Refs no store could resolve — i.e. the model made them up.
    unknown_refs: list[str] = Field(default_factory=list)

    @property
    def total_refs(self) -> int:
        return len(self.place_refs) + len(self.source_refs)

    @property
    def grounded(self) -> bool:
        """True when every cited ref resolves. An answer citing nothing is vacuously grounded."""
        return not self.unknown_refs


def extract_refs(text: str) -> tuple[list[str], list[str]]:
    """Return (place_refs, source_refs) found in ``text``, de-duplicated, in order."""
    return _unique(_PLACE_RE.findall(text)), _unique(_SOURCE_RE.findall(text))


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def verify_answer_refs(
    answer: str,
    *,
    place_store: PlaceStore | None = None,
    source_store: SourceStore | None = None,
) -> GroundingReport:
    """Check every ref cited in ``answer`` against the stores.

    A missing store means we cannot verify that kind of ref, so those refs are
    reported as unknown rather than quietly assumed valid — failing closed is the
    point of the check.
    """
    place_refs, source_refs = extract_refs(answer)
    unknown: list[str] = []

    for ref in place_refs:
        record = await place_store.get(ref) if place_store is not None else None
        if record is None:
            unknown.append(ref)

    for ref in source_refs:
        source = await source_store.get(ref) if source_store is not None else None
        if source is None:
            unknown.append(ref)

    return GroundingReport(place_refs=place_refs, source_refs=source_refs, unknown_refs=unknown)
