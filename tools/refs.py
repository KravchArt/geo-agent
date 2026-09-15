"""Handles — the anti-hallucination layer between the model and real-world data.

The problem
-----------
Coordinates (``55.7539, 37.6208``), long URLs and awkward proper nouns are
exactly the things an LLM copies wrong: a digit drops, a sign flips, a URL grows
a plausible-looking slug that never existed. And every such error is *silent* —
``55.7539`` and ``55.7593`` are both perfectly valid points, ~600 m apart.

The fix
-------
The model never handles that data at all. A tool that resolves something real
stores the full record in Redis and hands the model a short opaque **ref**:

    plc_a1b2c3d4e5   a place   (place:<ref>  in Redis -> PlaceRecord)
    src_9f8e7d6c5b   a web page (source:<ref> in Redis -> SourceRecord)

The model passes refs around between tools; adapters expand them back into
coordinates/URLs on the way out. So:

  * ``routing_tool(..., {"query": "Red Square", "area": "Moscow"})`` -> text resolution
  * ``places_search(mode=near, near="plc_a1b2c3d4e5")``
  * ``routing_tool(mode=rank, origins=["plc_a1b2c3d4e5"], ...)``

**The invariant: coordinates and URLs flow OUT of the system (to Redis, to the
UI, to the final answer) and NEVER back IN from the model.** That is why no
model-facing schema in ``tools/`` has a lat/lon or url input — and why the
outputs don't have them either: data the model cannot see, it cannot corrupt.

Refs are also the natural echo-grounding key: a ref in the answer is provably
backed by a real tool result, because only a tool can mint one.

Ref format
----------
``<prefix>_<10 lowercase hex>`` — short enough to copy without error, fixed
length (a truncated ref fails validation instead of resolving to something
else), and the prefix means a source ref passed where a place is expected is
rejected by the schema rather than misused.

The hex is derived from the IDENTITY of the resolved thing (the provider's
stable id/uri, or rounded coordinates), NOT from the query text — so "Red
Square" and "Red Square, Moscow" resolve to the same ref and hit the same cache
entry. Minting happens in the adapter (phase 1); this module only defines the
contract.

A ref that Redis no longer knows (expired, or invented by the model) must fail
the tool call with ``ToolErrorCode.UNKNOWN_REF`` — never be silently guessed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

#: Redis key namespaces. The records below live under `<ns>:<ref>`.
PLACE_NS = "place"
SOURCE_NS = "source"

PLACE_REF_PATTERN = r"^plc_[0-9a-f]{10}$"
SOURCE_REF_PATTERN = r"^src_[0-9a-f]{10}$"

PlaceRef = Annotated[str, StringConstraints(pattern=PLACE_REF_PATTERN)]
SourceRef = Annotated[str, StringConstraints(pattern=SOURCE_REF_PATTERN)]


def mint_place_ref(identity: str) -> PlaceRef:
    """Create a deterministic opaque place ref from provider-side identity."""

    normalized_identity = identity.strip()
    if not normalized_identity:
        raise ValueError("place identity cannot be empty")

    digest = hashlib.sha256(normalized_identity.encode("utf-8")).hexdigest()[:10]
    return f"plc_{digest}"


def mint_source_ref(identity: str) -> SourceRef:
    normalized = identity.strip()
    if not normalized:
        raise ValueError("source identity cannot be empty")

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return SourceRef(f"src_{digest}")


class RecordOrigin(StrEnum):
    """Where a cached record came from — kept for debugging and eval."""

    GEOCODE = "geocode"
    PLACES_SEARCH = "places_search"
    #: Minted by the orchestrator from the client's GPS fix, before the model runs.
    #: This is how "near me" works without the model ever seeing coordinates.
    USER_LOCATION = "user_location"


class GeoBounds(BaseModel):
    """Provider-supplied rectangular bounds kept outside model-facing schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _check_corner_order(self) -> GeoBounds:
        if self.west >= self.east:
            raise ValueError("west must be less than east")
        if self.south >= self.north:
            raise ValueError("south must be less than north")
        return self


class PlaceRecord(BaseModel):
    """Full truth about a place. Lives in Redis at ``place:<ref>``.

    This is the ONLY place coordinates exist between tool calls. Adapters read it
    to build ``ll``/``waypoints``; the map UI reads it to draw pins. The model
    never sees it — it only ever sees :attr:`ref` and :attr:`name`.
    """

    ref: PlaceRef
    name: str
    address: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    #: Toponym kind (locality/street/house/metro/district) — None for organisations.
    kind: str | None = None
    #: Geocoder match precision — None for organisations.
    precision: str | None = None
    #: Locality component parsed from the provider's structured address.
    locality: str | None = None
    #: Provider-resolved locality labels keyed by response language. These stay
    #: hidden from the model and let adapters compare translated municipality
    #: names without weakening geographic filtering to a rectangular bbox.
    localized_localities: dict[str, str] = Field(default_factory=dict)
    #: Provider-supplied viewport for a geocoded toponym.
    bounds: GeoBounds | None = None
    #: Adapter that supplied the record; internal provenance, never model-facing.
    provider: str | None = None
    #: Provider ids, when we have them (CompanyMetaData.id / GeocoderMetaData uri).
    provider_id: str | None = None
    provider_uri: str | None = None
    origin: RecordOrigin
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPlace:
    """A place ref paired with the hidden record used by downstream APIs."""

    ref: PlaceRef
    record: PlaceRecord


class SourceRecord(BaseModel):
    """Full truth about a web result. Lives in Redis at ``source:<ref>``.

    Same idea as :class:`PlaceRecord`, for URLs: the model cites ``src_...`` and
    the renderer expands it into a real link, so a fabricated URL is impossible
    by construction.
    """

    ref: SourceRef
    url: str
    title: str
    domain: str
    snippet: str
    published_date: date | None = None
    created_at: datetime | None = None
