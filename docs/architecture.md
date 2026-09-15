# Geo-Agent — technical stack and agent setup

A travel/location agent: an LLM that answers by calling geo and web tools, behind
gates that decide what it is allowed to work on, with every claim traceable back
to a real tool result.

---

## 1. Technical stack

### Runtime

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI** + uvicorn | async end to end; the pipeline is I/O-bound (model, tools, DB) |
| Contracts | **Pydantic v2** | one schema serves validation, the model's JSON Schema, and the API docs |
| Config | **pydantic-settings** | typed env config with startup guardrails |
| Database | **PostgreSQL** + SQLAlchemy 2.0 (async, asyncpg) | durable conversation transcript and observability; JSONB for sources, traces and tool payloads |
| Migrations | **Alembic** | schema changes are reviewable and repeatable |
| Cache/state | **Redis** | disposable session cache, refs and echo-grounding map; never the durable transcript source |
| LLM transport | **openai** SDK over httpx | any OpenAI-compatible backend: vLLM, Ollama, OpenRouter |
| HTTP | **httpx** (async) | one shared client for all tool providers |
| UI | **Streamlit** | thin debug client; separate image, not a backend dependency |
| Tooling | uv, ruff, mypy (strict), pytest | one lockfile; lint, types and tests gate every change |

No ML stack: both classifier gates call an OpenAI-compatible endpoint over
HTTP, so the install and the image carry no model weights.

### Layout

```
common/     shared contracts (LLM, ReAct, gates, API) — depends on nothing
tools/      tool schemas + provider adapters          — depends on common only
backend/    FastAPI app, pipeline, gates, orchestrator — depends on tools, common
ui/         Streamlit client                          — talks HTTP only
```

The dependency arrow never points backwards: `tools` cannot import `backend`. A
Redis-backed store that implements a `tools` protocol therefore lives in
`backend/app/stores.py`, not in `tools`.

The geo tools use the same rule internally:

```
tools/geo/
├── place_store.py       shared PlaceStore contract and in-memory implementation
├── geocoding/           contracts, matching, ref/address resolution, Yandex adapter
├── places_search/       anchor/scope orchestration, locality matching, POI adapters
└── routing/             input preparation, provider coordinator, route adapters
```

`geocoding/resolver.py` implements `GeocodedPlaceResolver` over the geocoder
contract, while `places_search` and `routing` depend on that public service.
The resolver owns only forward-geocoding policies for addresses, geographic
toponyms, and bounded localities; its small `ResolvedPlace` value object lives
in the same module. Provider-independent locality-label comparison lives in
`places_search/locality.py`, next to its only consumers.
`TextPlaceResolver` composes an ordered chain of city-scoped
`NamedPoiResolver` implementations with geocoding. Nearby-search anchors with a
city try 2GIS in covered agglomerations, then TomTom, then the geocoder; routing
retains its own geocoder-first policy. `PlacesSearchScopeResolver` replaces model-facing text
with one reusable ref before provider fallback begins, while
`PlacesSearchScopeLoader` only loads those prepared records inside concrete
providers. City and area boundaries always use the geocoder directly.
`place_store.py` stays at the neutral `geo` level because
geocoding, place search, and routing persist or load the same opaque records.

---

## 2. Request pipeline

```
POST /api/v1/chat  (the Streamlit UI also sends X-Client-ID)
  │
  ├─ persist request (status=running)
  ├─ load the last 10 transcript messages (Postgres; sole dialogue source)
  ├─ resolve user context
  │     ├─ browser geolocation (if granted)
  │     ├─ otherwise browser-resolved public-IP geolocation
  │     ├─ never reinterpret a UI proxy/container address as the user
  │     └─ determine local time from the resolved timezone
  │
  ├─ PREFLIGHT GATES ─ scope ∥ censorship (asyncio.gather)
  │     └─ gates evaluate the original user message
  │
  ├─ if allowed → ORCHESTRATOR (ReAct loop, ≤6 steps)
  │     model → tool_calls → executor → observation → model → … → answer
  │
  ├─ append user-context metadata only to the model-facing prompt
  ├─ OUTPUT CENSORSHIP — the same gate, re-run on the answer
  ├─ ECHO-GROUNDING — every cited ref must resolve; if not, one regeneration
  ├─ MAP SELECTION — the final response explicitly selects a subset of
  │                  this-turn `places_search` refs; the backend expands only
  │                  that subset into client coordinates
  │
  ├─ Postgres: durable conversation + messages and request observability
  └─ Redis: disposable session cache, echo:<hash>, place:/source:<ref>
```

A gate rejection is **HTTP 200** with `status=rejected`: refusing is a business
outcome, not a failure.

---

## 3. Gates

Both gates implement one interface (`GateEvaluator.evaluate`), so the pipeline
never knows which backend answered.

| Gate | Backends | Default |
|---|---|---|
| **scope** — "is this travel/geo at all?" | `rule_based` (15 regex rules) · `llm` (scoper prompt, bare yes/no) · `classifier` (fine-tuned e5 + `llm` inside its grey zone) | `rule_based` |
| **censorship** — "is this safe?" | `rule_based` (12 regex rules) · `model` (censor prompt, bare safe/unsafe) | `rule_based` |

**Scope sees the dialogue, censorship does not.** A follow-up like *"и что
рядом?"* is in scope only because of the previous turns, so the classifier gets
the last few turns as labelled context. Safety is judged on the message alone —
harmful intent must not become acceptable because the conversation opened
innocently.

**Every failure falls back to the regex gate.** A classifier that is down, times
out, or answers unparseably must not decide by default: silently allowing
everything is how out-of-scope traffic reaches the main model unnoticed.

Measured: for scope, a 0.5B model answers "yes" to everything (4/8 on the probe
set), 1.5B gets 6/8, **3B gets 8/8**. Do not go below 3B — a gate that always
passes is worse than no gate. The censorship prompt has no equivalent probe set
yet, so its false-positive rate on ordinary geography is unmeasured.

---

## 4. Orchestrator (the agent loop)

`backend/app/services/orchestrator.py` drives one request:

1. Send the conversation plus the **tool specs of the tools that actually exist**.
2. If the reply carries `tool_calls`, execute **all** of them (models emit
   parallel calls), append each result, and ask again.
3. Stop on a final answer or when the step budget (6) runs out.

Details that matter:

- **Real OpenAI tool protocol** — the assistant turn is replayed with its own
  `tool_calls` and each result as a `role="tool"` message keyed by
  `tool_call_id`. That is the shape tool-calling models are trained on;
  paraphrasing the turn as text makes them repeat calls.
- **Closed tool set** — an invented tool name never reaches the executor, and the
  model is told which names are real, so it self-corrects in one turn.
- **The system prompt is composed from the available tools.** The wire-level tool
  list already shrinks when a provider has no key; documenting a tool the model
  cannot call just invites a wasted turn.
- **Tool errors are data.** A failed call becomes an observation, not an
  exception — the model retries or explains honestly.
- **Conversation memory.** The last turns from the durable PostgreSQL transcript
  are replayed before the current message, so "which of them is closest?"
  resolves against the previous answer even after a UI/API/Redis restart.
  Redis session snapshots are never replayed into model context, preventing a
  guessed legacy ID or delete/cache race from resurrecting a transcript. Opaque
  refs are still Redis-backed and therefore have their own, shorter lifetime.
- **User-context metadata.** The pipeline appends a trusted metadata block to the
  end of the current user message before it is sent to the LLM. It contains the
  resolved location (browser geolocation when available, otherwise IP-based
  geolocation from a public address resolved in the browser) together with the
  user's local timezone and current local time. A missing/private UI address
  produces `unavailable`, never the server's egress location.
  This metadata is not stored as part of the conversation history and is not
  passed to the input gates, so scope and censorship continue to evaluate the
  original user message only.

---

## 5. Tools

| Tool | Provider | Needs |
|---|---|---|
| `places_search` | TomTom, 2GIS, OpenStreetMap, or Yandex | configured provider + geocoder key |
| `routing_tool` | 2GIS, Yandex, GraphHopper, or OSRM | configured routing provider + geocoder key |
| `web_search` | Exa, Tavily, Firecrawl | provider key |
| *geocoding* | TomTom | **internal** — the model never calls it |

Providers are enabled by list (`WEB_SEARCH_PROVIDERS=["exa"]`). Enabling one
without its key **fails at startup**, by design: a half-configured tool that
returns errors at runtime is worse than not booting.

`TEXT_PLACE_RESOLUTION_PROVIDERS=["twogis","tomtom"]` defines one ordered
named-place chain shared by `places_search` textual anchors and `routing_tool`
text points. Both consumers receive the same `TextPlaceResolver`; their separate
input adapters only translate their different schemas and error types. The shared
TomTom geocoder is the final fallback even when the named-place list is empty.

When multiple places providers are enabled, the coordinator uses the stable
2GIS, TomTom, OpenStreetMap, Yandex order and advances after an empty result or
a transient/provider response failure. Authentication, configuration, shared geocoder, and internal
contract failures are not hidden. Provider results are never merged: one
complete result set wins. Area and specifically named searches preserve the
provider's order after filtering and first-record deduplication; nearby searches
sort the surviving results by distance. Before provider attempts, the coordinator
resolves a textual city or nearby anchor to one opaque ref. Every fallback
provider reuses that ref and its hidden coordinates after both empty results and
retryable provider failures, avoiding duplicate geocoder or named-POI calls.
`places_search(mode="resolve")` accepts up to ten concrete organisation identities from
the user, conversation, or web evidence. It resolves them independently inside one shared
city area and returns one first-ranked, post-filtering/deduplication record per candidate,
or `not_found`/`error`. It serves singular entity lookup (for example, requesting the phone
of a named branch), not discovery of a chain's branches; that remains `area`/`near` work.
Web-only facts and their `src_` evidence remain outside the geo contract; successful
candidates receive normal `plc_` refs for routing.
TomTom POIs store their main entry point when available, so a returned ref is
better suited to subsequent routing than the POI's display-centre coordinate.
The 2GIS provider checks Regions Search before Places Search. A locality outside
2GIS detailed coverage returns an empty result without a Places API call, so the
coordinator advances to globally covered TomTom. For textual nearby anchors with
`city`, 2GIS Regions Search first checks whether the city, satellite, or settlement
belongs to a detailed 2GIS region. Covered anchors use 2GIS first; uncovered or
empty lookups continue to TomTom named-POI search and finally TomTom geocoding. The 2GIS resolver
validates the administrative locality, honours explicit object-type words from
the query, compares aliases from `name_ex`, and groups records representing the
same physical location. Repeated branches of a common brand remain separate during
discovery. Resolve mode retains the provider's first representative after name/address
filtering and deduplication. By contrast, a textual nearby anchor remains
clarification-sensitive: distinct surviving anchor groups produce selectable `plc_`
options rather than an automatic choice. A named distributed landmark can collapse
nearby entrances, stations, and component records while retaining the provider's first
representative. The model-facing schema therefore does not need an address-versus-POI
classifier.

2GIS, Yandex, OSRM, and GraphHopper are independent concrete routing providers. They
load coordinator-prepared refs through `RoutingPlaceResolver` and use shared
adapter functions for response validation, result materialization, matrix
ranking, warnings, and error mapping. The routing coordinator resolves all
unique textual points to opaque refs once before the first provider attempt.
Every provider therefore uses the same hidden coordinates during fallback
instead of repeating geocoding or named-POI resolution. OSRM supports the
configured driving and walking graphs, while GraphHopper also supports
bicycle/scooter profiles and waypoint optimization. Both expose static travel
times without live traffic. 2GIS is tried first when enabled: `route` uses
detailed Routing API legs with street names and navigation maneuvers (or Public
Transport API legs), and `rank` uses synchronous Distance Matrix batches.
GraphHopper is the first fallback.
Its route/matrix HTTP exchange has a strict 2-second deadline; input resolution,
geocoding, validation, and normalization are outside that provider-specific
deadline.

Fallback is an explicit policy rather than a broad error-code allowlist:
timeouts, network failures, rate limits, 5xx responses, and missing routes try
the next routing engine. A provider-local unsupported option also advances—for
example, `use_traffic=false` can skip 2GIS and use GraphHopper's static graph.
Invalid provider JSON/schema also falls back so the user can still receive a
route, but emits an error log for alerting. Authentication failures (401/403),
invalid refs, geocoding failures, and internal contract errors are returned
immediately because a successful secondary provider must not hide an operator
or code defect.

Routing transports decode provider JSON before classifying an HTTP error. This
preserves 2GIS per-route statuses, OSRM codes such as `NoSegment`, `NoRoute`,
`TooBig`, and `InvalidOptions`, GraphHopper error hints, and Yandex's documented
`errors` payload. Yandex exposes messages rather than a stable provider code, so those
responses keep `provider_code=null` instead of turning high-cardinality text
into a metric label. Every routing upstream-call metric records its outcome,
HTTP status, normalized error/failure kind, provider-specific code when one
exists, and retryability; therefore a successful fallback still retains the
failed first attempt.

`upstream_latency_ms` estimates the upstream critical path rather than blindly
adding every HTTP duration. Calls in one explicitly marked parallel batch
contribute their maximum latency; sequential calls and separate batches remain
additive. Routing geocoding uses batches of at most five text points, while OSM
reverse-geocoding groups its concurrent address enrichments. Provider fallback
attempts have distinct group IDs, so GraphHopper and OSRM latencies are never
mistaken for concurrent work.

### Refs — the anti-hallucination layer

Coordinates and URLs are exactly what an LLM copies wrong, and the error is
silent: `55.7539` and `55.7593` are both valid points ~600 m apart. So the model
never holds them.

A tool stores the full record in Redis and returns a short handle:

```
plc_a1b2c3d4e5   a place    (place:<ref>  → PlaceRecord with lat/lon)
src_9f8e7d6c5b   a web page (source:<ref> → SourceRecord with the URL)
```

The model passes refs between tools; geo tool implementations load their hidden
records before calling upstream providers.
**Data flows out to Redis and the UI, never back in from the model.** Refs are
fixed-length (a truncated one fails validation instead of resolving to something
else) and prefixed, so a source ref passed where a place is expected is a schema
error.

The model returns ordinary user-facing Markdown and appends an inline
`[[plc_...]]` marker beside every concrete place selected for the answer. These
markers are removed from the public stream and final text. Before publishing,
the backend gives the model one rewrite when any referenced `plc_` or `src_`
record does not exist. For map pins it additionally accepts only refs present in
a successful `places_search` result from the current request, expands their real
coordinates, and returns them as `map.places`. The expanded pins are saved with
the assistant message, so reopening history does not depend on the Redis ref TTL.

### Conversation state and browser identity

PostgreSQL is the source of truth for user-visible history. `conversation`
stores the owner, title and timestamps for a `session_id`;
`conversation_message` stores chronological user/assistant messages together
with their sources, terminal status and optional rejection reason. Assistant
rows also retain resolved `places_search.area` values in a private `search_areas`
field. The field is omitted from the history API and is injected only into model
context, allowing a follow-up area search to reuse `area_ref` instead of resolving
the same city again. The pipeline reads the last 10 durable messages for gates
and model context, while the detail endpoint returns the complete transcript.

The history API is scoped by the required `X-Client-ID` request header:

| Endpoint | Result |
|---|---|
| `GET /api/v1/conversations?limit=50&offset=0` | `{items, limit, offset}`; summaries contain `session_id`, title, timestamps and message count |
| `GET /api/v1/conversations/{session_id}` | one summary plus chronological `{id, role, content, sources, status, rejection_reason, created_at}` messages |
| `DELETE /api/v1/conversations/{session_id}` | delete the owned user-visible transcript; returns `204` |

An unknown conversation and a conversation owned by another client both return
`404`, so the API does not reveal whether another browser's identifier exists.
`POST /api/v1/chat` and `POST /api/v1/chat/stream` accept the same header and the
UI always sends it. The header remains optional for older API callers: their
transcript is durable and usable for follow-ups by `session_id`, but an unowned
conversation is not exposed by the list/detail/delete API. Clients that need
sidebar history must therefore send the same header from their first chat turn;
an already-owned mismatch is rejected before the LLM is called.

Conversation deletion cascades to `conversation_message`, but it is not a purge
of the separate execution audit trail: `request`, model/tool traces and metrics
follow their own retention policy. A deployment offering privacy erasure must
delete or anonymize those records as a separate operation.

The Streamlit UI keeps an anonymous browser UUID under
`localStorage["geoagent_client_id"]` and sends it as `X-Client-ID`. The active
conversation is a separate UUID in the URL as `?conversation=<uuid>`. Those two
browser-persistent values replace `st.session_state` as the identity source:

1. **Refresh/reopen:** recover the browser ID and active UUID, then load list and
   detail from the API. Refresh does not create a different conversation.
2. **New conversation:** put a fresh UUID in `conversation`, clear the rendered
   transcript, and create no database row until the stream returns a terminal
   `result`.
3. **Select:** replace the URL UUID with the selected sidebar entry and fetch its
   transcript.
4. **Delete:** call the owned delete endpoint, remove the entry from the sidebar,
   and move to a fresh empty UUID if the active conversation was deleted. The
   backend also removes the known `session:<id>:*` cache keys on a best-effort
   basis, while global content-addressed refs remain untouched.

The browser ID is useful anonymous namespacing, **not authentication**. Anyone
who obtains it can present the same header; it provides neither verified user
identity, cross-device history nor account recovery. Clearing site data creates
a new browser ID and makes the old owned history unreachable from that browser.
A multi-user deployment must replace or bind it with server-verified login and
authorization.

Redis remains disposable cache/ref infrastructure:

| Key | Contents | TTL |
|---|---|---|
| `session:<id>:history` | legacy pre-0005 key; no longer written or replayed | `REDIS_SESSION_TTL` (1 h) |
| `session:<id>:last_*` | snapshot of the latest request | `REDIS_SESSION_TTL` (1 h) |
| `place:<ref>` / `source:<ref>` | the real data behind opaque refs | `REDIS_REF_TTL` (24 h) |
| `echo:<tool_hash>` | tool result, for after-the-fact proof | `REDIS_ECHO_TTL` (30 min) |

Losing or deleting Redis state does not delete the transcript or conversation
list. It does discard scratch state and may make an old `plc_`/`src_` handle
unresolvable after its TTL; the user-visible text remains durable in Postgres.
Refs and echo are not session-scoped and outlive the short session cache
deliberately.

### Echo-grounding

Every tool result is stored under `echo:<tool_hash>` (sha256 of tool name +
canonical args), and every ref cited in the final answer is resolved against the
stores. An unresolvable ref was invented: the model gets **one regeneration**
with the offending refs named, reusing the tool results already in context.

Fails closed — a missing store means "unverified", not "assumed fine".

---

## 6. Configuration

Everything is env-driven (`backend/app/config.py`), `.env` is git-ignored.

```bash
LLM_MODE=vllm                    # mock | vllm (any OpenAI-compatible endpoint)
LLM_BASE_URL=...                 # vLLM, Ollama or OpenRouter
LLM_MODEL=qwen/qwen3-30b-a3b-instruct-2507
LLM_HTTP_PROXY=                  # explicit; env proxies are ignored on purpose
SCOPE_PROVIDER=rule_based        # rule_based | llm | classifier | no_scoper
CENSORSHIP_PROVIDER=rule_based   # rule_based | model (+ CENSORSHIP_BASE_URL/MODEL)
WEB_SEARCH_PROVIDERS=["exa"]
TOOLS_HTTP_PROXY=
```

**Proxies are configured, never inherited.** `HTTP_PROXY`/`ALL_PROXY` from the
shell are deliberately ignored (`trust_env=False` everywhere): routing is a
deployment decision, and an ambient `socks://` value would stop httpx from
starting at all. Tests clear the variables in the root `conftest.py`, so the
suite behaves the same on every machine.

---

## 7. Observability

Durable product data in Postgres:

- `conversation` — browser owner, title and list-order timestamps
- `conversation_message` — ordered public transcript, sources, public map pins,
  private reusable search-area refs and terminal result metadata

Per request, in Postgres:

- `request` — original request and final status
- `react_trace` — complete ReAct trace (reasoning, tool calls, observations, grounding)
- `pipeline_stage_metric` — latency and status of every pipeline stage
- `gate_check_log` — every scope/censorship gate execution, provider, latency and outcome
- `llm_call_metric` — every LLM call, latency, tokens and outcome
- `tool_call` — every tool invocation, arguments, latency and upstream-call count
- `upstream_call_metric` — every external provider request, latency, HTTP status and outcome
- `metrics` — aggregated per-request statistics (overall latency, stage latencies, call counts, tokens, regenerations)

Relationships:

- `conversation_id` links every transcript message to its conversation; deleting
  the conversation cascades to its messages
- optional `request_id` links a transcript message back to the execution that
  produced it without making the transcript depend on the audit row's lifetime
- `request_id` links `request`, `metrics`, `pipeline_stage_metric`, `gate_check_log` and `react_trace`
- `trace_id` links `react_trace` with `llm_call_metric` and `tool_call`
- `tool_call_id` links `tool_call` with `upstream_call_metric`

A ready-to-use SQL report is available in
`scripts/sql/observability_summary.sql`. It aggregates the complete execution
of every request, including total latency, stage latencies, gate timings, LLM
calls, tool calls, upstream calls, token usage and failures.

---

## 8. Known gaps

- **Refs vs. the prompt.** The orchestrator prompt says place refs are internal
  and should not be shown to the user, while echo-grounding verifies refs *in the
  answer*. When the model writes prose without refs, the check is vacuous. Either
  refs belong in the answer (and the renderer expands them), or grounding must
  verify something else.
- **Ungrounded answers still ship.** After the one regeneration, an answer with
  invented refs is returned and recorded, not blocked. Blocking is a policy
  decision, left open.
- **The censorship prompt is unmeasured.** `LLMCensorshipGate` mirrors the scope
  gate, but there is no probe set for it the way `test_scope_gate_llm.py` covers
  scope, so the false-positive rate on ordinary geography is unknown.
