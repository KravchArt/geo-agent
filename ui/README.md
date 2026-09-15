# Streamlit UI — Phase 0

The UI sends one chat message to `POST /api/v1/chat/stream`, renders live gate,
ReAct and tool progress plus `answer_delta` text events from the NDJSON stream,
then displays the terminal shared LLM/ReAct response contract. `answer_reset`
discards the visible draft when it was tool-call narration, needs regeneration,
or fails output censorship; the terminal result then supplies the safe answer.
The live progress row is informational rather than expandable. Scope and
censorship checks are shown as separate steps before route selection.
The non-streaming `POST /api/v1/chat` endpoint remains available for API clients.
In Docker it uses `API_URL=http://api:8000`.
For host-side development use `API_URL=http://localhost:8000`.

## Conversation identity and history

On its first load the UI atomically reads or creates an anonymous browser UUID in
the `geoagent_client_id` localStorage key. It sends that value in the
`X-Client-ID` header with chat and conversation-history requests. Clearing site
storage creates a new browser identity and therefore an empty history; this UUID
is device-local identification, not user authentication.

The active conversation UUID is stored in the `conversation` URL query parameter.
A full page refresh therefore reconnects to the same conversation and reloads its
messages, sources included, from the backend. The sidebar uses:

- `GET /api/v1/conversations?limit=50&offset=0` for the newest-first list;
- `GET /api/v1/conversations/{session_id}` for the active transcript;
- `DELETE /api/v1/conversations/{session_id}` after explicit confirmation.

Clicking **Новый диалог** creates only a fresh client-side UUID and an
empty transcript. It does not appear in history until the first completed agent
response has been persisted. If the backend does not expose `DELETE`, the rest of
the history UI remains usable and reports that deletion is unavailable.

## Location context

The location toggle uses the browser Geolocation API through
`streamlit-js-eval`. For host-side UI development install the UI dependencies:

```bash
pip install pandas==2.2.3 streamlit==1.53.0 httpx==0.28.1 streamlit-js-eval==1.0.0
```

When precise location is disabled or denied, the UI obtains the public address
from the browser through `UI_PUBLIC_IP_LOOKUP_URL` (ipify by default) and sends
only that address to the backend for approximate IP-based location. This avoids
mistaking a Docker bridge, reverse proxy, SSH tunnel, or backend egress address
for the user. `st.context.ip_address` is used only when it is already a public
address and the browser lookup is unavailable. The browser timezone is always
sent when available.

## Map pins

Each assistant answer renders its own map when the model explicitly selected
places in that answer. The backend expands those verified `plc_` refs into
coordinates; the UI never parses place names or coordinates from Markdown.
Earlier answer maps remain in the transcript, while an enabled browser location
is shown alongside the selected places on each relevant map.

To preview the layout without API, Redis, PostgreSQL, or an LLM, run without
adding UI-only packages to the project environment:

```bash
uv run --with streamlit==1.53.0 --with pandas==2.2.3 --with pydeck==0.9.1 streamlit run ui/map_preview.py
```
