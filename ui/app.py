from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st
from streamlit_js_eval import get_geolocation, streamlit_js_eval

from ui.client_ip import normalize_public_ip
from ui.map_view import MapMarker, render_map
from ui.markdown import prepare_chat_markdown

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PUBLIC_IP_LOOKUP_URL = os.getenv(
    "UI_PUBLIC_IP_LOOKUP_URL", "https://api64.ipify.org?format=json"
).strip()
# An agent turn is a tool loop, not one completion: several model calls plus the
# tool round-trips. A small model on CPU needs minutes, so the old 30s cut every
# real answer short.
REQUEST_TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "300"))
HISTORY_TIMEOUT = float(os.getenv("UI_HISTORY_TIMEOUT", "10"))
CLIENT_ID_HEADER = "X-Client-ID"
CONVERSATION_QUERY_PARAM = "conversation"
CLIENT_ID_STORAGE_KEY = "geoagent_client_id"
BROWSER_ID_ERROR_PREFIX = "geoagent_browser_storage_error:"

# One expression performs the read and conditional write, so two component
# reruns can never replace an already-created browser id. randomUUID is not
# available in every non-secure browser context, hence the getRandomValues
# fallback.
_BROWSER_ID_EXPRESSION = f"""
(() => {{
  try {{
    const storageKey = {json.dumps(CLIENT_ID_STORAGE_KEY)};
    const uuidPattern = new RegExp(
      "^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[1-5][0-9a-f]{{3}}-" +
      "[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$",
      "i"
    );
    let clientId = window.localStorage.getItem(storageKey);
    if (!clientId || !uuidPattern.test(clientId)) {{
      if (typeof window.crypto.randomUUID === "function") {{
        clientId = window.crypto.randomUUID();
      }} else {{
        const bytes = window.crypto.getRandomValues(new Uint8Array(16));
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = Array.from(bytes, value => value.toString(16).padStart(2, "0"));
        clientId = [
          hex.slice(0, 4).join(""),
          hex.slice(4, 6).join(""),
          hex.slice(6, 8).join(""),
          hex.slice(8, 10).join(""),
          hex.slice(10).join("")
        ].join("-");
      }}
      window.localStorage.setItem(storageKey, clientId);
    }}
    return clientId;
  }} catch (error) {{
    const name = error && error.name ? error.name : "unknown";
    return {json.dumps(BROWSER_ID_ERROR_PREFIX)} + name;
  }}
}})()
"""

# Resolve the address from the browser, not from the UI/API containers. On a
# remote Docker deployment their socket peer is commonly a bridge, reverse
# proxy, or SSH tunnel; asking an IP-geolocation provider from the backend then
# locates the server's egress address instead of the user.
_BROWSER_PUBLIC_IP_EXPRESSION = f"""
(() => {{
  const url = {json.dumps(PUBLIC_IP_LOOKUP_URL)};
  if (!url) return null;
  return fetch(url, {{cache: "no-store"}})
    .then(response => {{
      if (!response.ok) throw new Error("public IP lookup failed");
      return response.json();
    }})
    .then(payload => typeof payload.ip === "string" ? payload.ip : null)
    .catch(() => null);
}})()
"""


def _client(timeout: float) -> httpx.Client:
    """The API is reached directly — never through a proxy.

    httpx reads HTTP_PROXY/ALL_PROXY from the environment unless told not to, so
    on a machine behind a proxy the UI would try to tunnel its own localhost call
    (and an ``ALL_PROXY=socks://...`` makes it refuse to build a client at all).
    """
    return httpx.Client(timeout=timeout, trust_env=False)


def _client_headers(client_id: str) -> dict[str, str]:
    return {CLIENT_ID_HEADER: client_id}


def _uuid_text(value: object) -> str | None:
    """Return a canonical UUID string, rejecting absent or malformed browser input."""
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return None


def _progress_line(event: dict[str, object]) -> str:
    """Turn one backend progress event into a compact, persistent UI line."""
    stage = event.get("stage")
    status = event.get("status")
    message = str(event.get("message") or "Working…")
    if stage == "tool":
        icon = "🔧" if status == "running" else ("✅" if status == "completed" else "⚠️")
        if status == "running" and event.get("tool_input"):
            arguments = json.dumps(event["tool_input"], ensure_ascii=False, separators=(",", ":"))
            if len(arguments) > 240:
                arguments = arguments[:237] + "…"
            return f"{icon} {message} `{arguments}`"
        return f"{icon} {message}"
    if stage == "react":
        return f"🧠 {message}"
    if stage == "scope":
        return f"🎯 Scope · {message}"
    if stage == "censorship":
        return f"🛡️ Censor · {message}"
    if stage == "preflight":
        return f"🛡️ {message}"
    if stage == "routing":
        return f"↪️ {message}"
    if stage == "grounding":
        return f"🔎 {message}"
    if stage == "persistence":
        return f"💾 {message}"
    return message


def _render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    st.markdown("#### Источники")
    for source in sources:
        title = str(source.get("title") or source.get("domain") or "Источник")
        url = str(source.get("url") or "")
        domain = str(source.get("domain") or "")
        if url:
            st.markdown(f"- [{title}]({url}){' · ' + domain if domain else ''}")


def _conversation_summaries(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Conversation list has an unexpected response shape")
    return [item for item in payload["items"] if isinstance(item, dict)]


def _conversation_messages(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError("Conversation detail has an unexpected response shape")

    messages: list[dict[str, object]] = []
    for raw_message in payload["messages"]:
        if not isinstance(raw_message, dict):
            continue
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        raw_sources = raw_message.get("sources", [])
        sources = (
            [source for source in raw_sources if isinstance(source, dict)]
            if isinstance(raw_sources, list)
            else []
        )
        raw_map = raw_message.get("map")
        map_data = raw_map if isinstance(raw_map, dict) else None
        messages.append(
            {
                "role": role,
                "content": content,
                "sources": sources,
                "map": map_data,
                "status": raw_message.get("status"),
                "rejection_reason": raw_message.get("rejection_reason"),
            }
        )
    return messages


def _get_conversations(client_id: str) -> list[dict[str, object]]:
    with _client(HISTORY_TIMEOUT) as client:
        response = client.get(
            f"{API_URL}/api/v1/conversations",
            headers=_client_headers(client_id),
            params={"limit": 50, "offset": 0},
        )
        response.raise_for_status()
        return _conversation_summaries(response.json())


def _get_conversation(conversation_id: str, client_id: str) -> list[dict[str, object]] | None:
    """Load a transcript; None means the UUID is a new, not-yet-persisted chat."""
    with _client(HISTORY_TIMEOUT) as client:
        response = client.get(
            f"{API_URL}/api/v1/conversations/{conversation_id}",
            headers=_client_headers(client_id),
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return _conversation_messages(response.json())


def _delete_conversation(conversation_id: str, client_id: str) -> str:
    """Return deleted, missing or unsupported; raise for operational failures."""
    with _client(HISTORY_TIMEOUT) as client:
        response = client.delete(
            f"{API_URL}/api/v1/conversations/{conversation_id}",
            headers=_client_headers(client_id),
        )
        if response.status_code in {httpx.codes.OK, httpx.codes.NO_CONTENT}:
            return "deleted"
        if response.status_code == httpx.codes.NOT_FOUND:
            return "missing"
        if response.status_code in {
            httpx.codes.METHOD_NOT_ALLOWED,
            httpx.codes.NOT_IMPLEMENTED,
        }:
            return "unsupported"
        response.raise_for_status()
    raise RuntimeError("Conversation deletion returned an unexpected response")


def _new_conversation() -> None:
    conversation_id = str(uuid.uuid4())
    st.session_state.session_id = conversation_id
    st.session_state.messages = []
    st.session_state.loaded_conversation_id = conversation_id
    st.session_state.pending_delete = None
    st.session_state.conversation_load_error = None
    st.session_state.chat_error = None
    st.query_params[CONVERSATION_QUERY_PARAM] = conversation_id


def _select_conversation(conversation_id: str) -> None:
    st.session_state.pending_delete = None
    st.query_params[CONVERSATION_QUERY_PARAM] = conversation_id
    # Loading is performed at the top of the next run, from the canonical URL.
    st.session_state.loaded_conversation_id = None


def _display_title(summary: dict[str, object]) -> str:
    title = str(summary.get("title") or "Новый диалог").strip() or "Новый диалог"
    return title if len(title) <= 42 else title[:39].rstrip() + "…"


st.set_page_config(page_title="GeoAgent — travel assistant", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoAgent")
# What the user gets, not how it is wired. The pipeline details live in the
# sidebar and the response expanders, where someone debugging will look.
st.caption("Ask about places, routes and trips — answers are grounded in real map and web data.")

browser_id_result = streamlit_js_eval(
    js_expressions=_BROWSER_ID_EXPRESSION,
    key="geoagent_browser_id",
)
if isinstance(browser_id_result, str) and browser_id_result.startswith(BROWSER_ID_ERROR_PREFIX):
    st.error("Не удалось получить доступ к локальному хранилищу браузера.")
    st.caption(
        "Разрешите хранение данных для этого сайта и перезагрузите страницу — "
        "оно нужно, чтобы безопасно привязать историю к этому браузеру."
    )
    st.stop()
browser_id = _uuid_text(browser_id_result)
if browser_id is None:
    with st.spinner("Восстанавливаем историю диалогов…"):
        st.stop()

if "browser_location" not in st.session_state:
    st.session_state.browser_location = None
if "browser_public_ip" not in st.session_state:
    st.session_state.browser_public_ip = None
if "location_enabled" not in st.session_state:
    st.session_state.location_enabled = False
if "request_in_flight" not in st.session_state:
    st.session_state.request_in_flight = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

browser_public_ip_result = streamlit_js_eval(
    js_expressions=_BROWSER_PUBLIC_IP_EXPRESSION,
    key="geoagent_browser_public_ip",
)
resolved_browser_public_ip = normalize_public_ip(browser_public_ip_result)
if resolved_browser_public_ip is not None:
    st.session_state.browser_public_ip = resolved_browser_public_ip

# The browser lookup wins because it observes the user's actual internet
# egress. The Streamlit connection address remains a no-network fallback for
# direct deployments where it is already public.
client_public_ip = normalize_public_ip(st.session_state.browser_public_ip) or normalize_public_ip(
    st.context.ip_address
)

requested_conversation_id = _uuid_text(st.query_params.get(CONVERSATION_QUERY_PARAM))
if requested_conversation_id is None:
    requested_conversation_id = str(uuid.uuid4())
    st.query_params[CONVERSATION_QUERY_PARAM] = requested_conversation_id

if st.session_state.get("session_id") != requested_conversation_id:
    st.session_state.session_id = requested_conversation_id
    st.session_state.loaded_conversation_id = None
    st.session_state.messages = []
    st.session_state.pending_delete = None

if st.session_state.get("loaded_conversation_id") != requested_conversation_id:
    try:
        stored_messages = _get_conversation(requested_conversation_id, browser_id)
        st.session_state.messages = stored_messages or []
        st.session_state.loaded_conversation_id = requested_conversation_id
        st.session_state.conversation_load_error = None
    except Exception as exc:
        # Keep retrying on later reruns; an API outage must not silently mark an
        # empty transcript as successfully loaded.
        st.session_state.messages = []
        st.session_state.conversation_load_error = str(exc)

try:
    conversation_summaries = _get_conversations(browser_id)
    history_error = None
except Exception as exc:
    conversation_summaries = []
    history_error = str(exc)
with st.sidebar:
    st.subheader("Status")
    try:
        with _client(5.0) as client:
            health = client.get(f"{API_URL}/health").json()
            info = client.get(f"{API_URL}/").json()
        if health.get("status") == "ok":
            st.success("Backend healthy")
        else:
            st.warning("Backend degraded")
        checks = health.get("checks", {})
        st.caption(
            f"Postgres {'✅' if checks.get('postgres') else '❌'} · "
            f"Redis {'✅' if checks.get('redis') else '❌'} · "
            f"LLM `{info.get('llm_mode', '?')}`"
        )
        with st.expander("Raw health"):
            st.json(health)
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")

    st.subheader("Location context")
    st.session_state.location_enabled = st.toggle(
        "Use precise browser location",
        value=st.session_state.location_enabled,
        help=(
            "Your browser will ask for permission. Coordinates are sent only "
            "with chat requests.\n"
            "ЧТОБЫ РАБОТАЛО ПИШИ В АДРЕСНОЙ СТРОКЕ ВМЕСТО 0.0.0.0 "
            "localhost или 127.0.0.1"
        ),
    )
    if st.session_state.location_enabled:
        location = get_geolocation()
        if location and "error" in location:
            error = location["error"]
            if error.get("code") == 1:
                st.warning("Location permission was denied; IP-based location will be used.")
            else:
                st.warning(
                    f"Could not read browser location: {error.get('message', 'unknown error')}"
                )
        elif location and location.get("coords"):
            coords = location["coords"]
            st.session_state.browser_location = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "accuracy_m": coords.get("accuracy"),
            }
            st.success("Precise location is enabled")
            st.caption(
                f"{coords['latitude']:.5f}, {coords['longitude']:.5f} "
                f"(±{coords.get('accuracy', 0):.0f} m)"
            )
    else:
        st.session_state.browser_location = None
        if client_public_ip is not None:
            st.caption("Precise location is off; approximate browser-IP location is available.")
        else:
            st.caption("Precise location is off; approximate IP location is unavailable.")
    st.caption(f"Timezone: `{st.context.timezone or 'UTC'}`")

    # Conversation controls deliberately sit immediately below location context.
    st.subheader("История диалогов")
    if st.button(
        "➕ Новый диалог",
        use_container_width=True,
        disabled=st.session_state.request_in_flight,
    ):
        _new_conversation()
        st.rerun()

    flash_message = st.session_state.pop("history_flash", None)
    if flash_message:
        st.success(str(flash_message))
    if history_error:
        st.warning(f"История временно недоступна: {history_error}")
    elif not conversation_summaries:
        st.caption("Сохранённых диалогов пока нет.")

    for summary in conversation_summaries:
        conversation_id = _uuid_text(summary.get("session_id"))
        if conversation_id is None:
            continue
        title = _display_title(summary)
        is_active = conversation_id == st.session_state.session_id
        display_title = f"✓ {title}" if is_active else title
        select_column, delete_column = st.columns([7, 1], gap="small")
        if select_column.button(
            display_title,
            key=f"select_conversation_{conversation_id}",
            type="secondary",
            use_container_width=True,
            disabled=st.session_state.request_in_flight,
        ):
            _select_conversation(conversation_id)
            st.rerun()
        if delete_column.button(
            "×",
            key=f"request_delete_conversation_{conversation_id}",
            help=f"Удалить «{title}»",
            use_container_width=True,
            disabled=st.session_state.request_in_flight,
        ):
            st.session_state.pending_delete = conversation_id

    pending_delete = _uuid_text(st.session_state.pending_delete)
    if pending_delete is not None:
        pending_summary = next(
            (
                summary
                for summary in conversation_summaries
                if _uuid_text(summary.get("session_id")) == pending_delete
            ),
            None,
        )
        pending_title = _display_title(pending_summary or {})
        st.warning(f"Точно удалить «{pending_title}»?")
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Удалить",
            key="confirm_conversation_delete",
            type="primary",
            use_container_width=True,
        ):
            try:
                delete_result = _delete_conversation(pending_delete, browser_id)
                if delete_result == "unsupported":
                    st.error("Удаление пока не поддерживается backend.")
                else:
                    st.session_state.pending_delete = None
                    st.session_state.history_flash = (
                        "Диалог удалён из видимой истории; технические журналы "
                        "сохраняются по отдельной политике."
                    )
                    if pending_delete == st.session_state.session_id:
                        _new_conversation()
                    st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить диалог: {exc}")
        if cancel_column.button(
            "Отмена",
            key="cancel_conversation_delete",
            use_container_width=True,
        ):
            st.session_state.pending_delete = None
            st.rerun()

    st.caption("Session")
    st.code(st.session_state.session_id, language=None)


def _render_chat_interface() -> None:
    conversation_load_error = st.session_state.get("conversation_load_error")
    if conversation_load_error:
        st.warning(f"Не удалось загрузить диалог: {conversation_load_error}")
        # Any Streamlit interaction starts another run and retries the load.
        st.button("Повторить загрузку", key="retry_conversation_load")
    chat_error = st.session_state.pop("chat_error", None)
    if chat_error:
        st.error(f"Pipeline request failed: {chat_error}")

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(prepare_chat_markdown(item["content"]))
            if item["role"] == "assistant":
                _render_sources(item.get("sources", []))
                _render_message_map(item)
                if item.get("status") == "rejected":
                    st.warning(f"Request rejected: {item.get('rejection_reason')}")
            if item["role"] == "assistant" and item.get("progress"):
                with st.expander("ReAct steps"):
                    for line in item["progress"]:
                        st.markdown(line)
            if item["role"] == "assistant":
                scope_gate = item.get("scope_gate")
                censorship_gate = item.get("censorship_gate")

                if scope_gate is not None or censorship_gate is not None:
                    with st.expander("Preflight gate decisions"):
                        if scope_gate is not None:
                            st.markdown("**Scope Gate**")
                            st.json(scope_gate)
                        if censorship_gate is not None:
                            st.markdown("**Censorship Gate**")
                            st.json(censorship_gate)
    submitted_prompt = st.chat_input(
        "Agent is working now😎"
        if st.session_state.request_in_flight
        else "Find a hotel near the British Museum",
        disabled=st.session_state.request_in_flight or bool(conversation_load_error),
    )
    if submitted_prompt:
        st.session_state.messages.append({"role": "user", "content": submitted_prompt})
        st.session_state.pending_prompt = submitted_prompt
        st.session_state.request_in_flight = True
        # Render the disabled input before starting the blocking streaming request.
        st.rerun()

    prompt = st.session_state.pending_prompt if st.session_state.request_in_flight else None
    if prompt is None:
        return
    try:
        data = None
        progress_lines: list[str] = []
        streamed_answer = ""
        request_json = {
            "session_id": st.session_state.session_id,
            "message": prompt,
            "user_context": {
                "browser_location": st.session_state.browser_location,
                "timezone": st.context.timezone or "UTC",
                "ip_address": client_public_ip,
            },
        }
        with st.chat_message("assistant"):
            # Keep operational activity in one fixed row above the answer. A
            # status widget draws an expander chevron even when it has no body,
            # so use a replaceable info message that is visibly non-interactive.
            progress_box = st.empty()
            progress_box.info("⏳ Starting request…")
            answer_placeholder = st.empty()
            with _client(REQUEST_TIMEOUT) as client:
                for attempt in range(2):
                    request_json["session_id"] = st.session_state.session_id
                    with client.stream(
                        "POST",
                        f"{API_URL}/api/v1/chat/stream",
                        json=request_json,
                        headers=_client_headers(browser_id),
                    ) as response:
                        if response.status_code == httpx.codes.NOT_FOUND and attempt == 0:
                            # A stale/shared URL can name a conversation owned by
                            # another browser. Rotate transparently and retry the
                            # user's first message instead of leaving a dead chat.
                            _new_conversation()
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            recovery_line = "♻️ Started a fresh conversation for this browser."
                            progress_lines.append(recovery_line)
                            progress_box.info("♻️ Starting a fresh conversation…")
                            continue

                        response.raise_for_status()
                        for raw_line in response.iter_lines():
                            if not raw_line.strip():
                                continue
                            event = json.loads(raw_line)
                            if event.get("type") == "progress":
                                line = _progress_line(event)
                                progress_lines.append(line)
                                progress_box.info(line)
                            elif event.get("type") == "answer_delta":
                                streamed_answer += str(event.get("delta") or "")
                                answer_placeholder.markdown(
                                    prepare_chat_markdown(streamed_answer) + "▌"
                                )
                            elif event.get("type") == "answer_reset":
                                streamed_answer = ""
                                answer_placeholder.empty()
                            elif event.get("type") == "heartbeat":
                                progress_box.info("⏳ The model is still working…")
                            elif event.get("type") == "result":
                                data = event["data"]
                            elif event.get("type") == "error":
                                raise RuntimeError(event.get("message", "Pipeline stream failed"))
                    break

            if data is None:
                raise RuntimeError("Pipeline stream ended without a result")

            answer = data["answer"]
            answer_placeholder.markdown(prepare_chat_markdown(answer))
            _render_sources(data.get("sources", []))
            progress_box.success(
                f"Completed · {len(progress_lines)} progress events",
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "progress": progress_lines,
                    "sources": data.get("sources", []),
                    "map": data.get("map"),
                    "status": data.get("status"),
                    "rejection_reason": data.get("rejection_reason"),
                    "scope_gate": data.get("scope_gate"),
                    "censorship_gate": data.get("censorship_gate"),
                }
            )
            # The sidebar was rendered before this terminal result existed.
            # Rerunning reloads its durable summary immediately; the response and
            # progress details above are retained in session_state and re-rendered.
            st.rerun()
    except Exception as exc:
        # Keep the optimistic user turn visible. In particular, a 409 means an
        # earlier request still owns this session; replacing local state with
        # the durable transcript made the newly typed message simply vanish.
        # A page reload or conversation re-selection still reconciles with the
        # database after ambiguous network failures.
        st.session_state.chat_error = str(exc)
    finally:
        st.session_state.pending_prompt = None
        st.session_state.request_in_flight = False
        st.rerun()


def _coordinate(value: object, *, minimum: float, maximum: float) -> float | None:
    """Validate external API data again before handing it to the map component."""

    if not isinstance(value, (int, float)):
        return None
    coordinate = float(value)
    return coordinate if minimum <= coordinate <= maximum else None


def _render_message_map(message: dict[str, object]) -> None:
    """Render the verified pins belonging to one assistant response."""

    map_data = message.get("map")
    if not isinstance(map_data, dict):
        return
    raw_places = map_data.get("places")
    if not isinstance(raw_places, list):
        return
    selected_places = [place for place in raw_places if isinstance(place, dict)]
    markers: list[MapMarker] = []
    has_search_anchor = False

    for place in selected_places:
        latitude = _coordinate(place.get("latitude"), minimum=-90, maximum=90)
        longitude = _coordinate(place.get("longitude"), minimum=-180, maximum=180)
        if latitude is None or longitude is None:
            continue
        name = str(place.get("name") or "Место")
        address = str(place.get("address") or "")
        is_search_anchor = place.get("marker_role") == "search_anchor"
        has_search_anchor = has_search_anchor or is_search_anchor
        markers.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "label": f"Точка поиска: {name}" if is_search_anchor else name,
                "address": address,
                "color": [37, 99, 235, 230] if is_search_anchor else [220, 38, 38, 220],
                "radius_m": 65 if is_search_anchor else 50,
            }
        )

    # A browser location supplements answer pins; it must not create a map for
    # an answer which did not select any valid places of its own.
    if not markers:
        return

    browser_location = st.session_state.browser_location
    if not has_search_anchor and isinstance(browser_location, dict):
        latitude = _coordinate(browser_location.get("latitude"), minimum=-90, maximum=90)
        longitude = _coordinate(browser_location.get("longitude"), minimum=-180, maximum=180)
        if latitude is not None and longitude is not None:
            markers.insert(
                0,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "label": "Ваше местоположение",
                    "address": "Точная геолокация из браузера",
                    "color": [37, 99, 235, 230],
                    "radius_m": 65,
                },
            )

    st.markdown("#### Карта")
    render_map(markers, height=360)


_render_chat_interface()
