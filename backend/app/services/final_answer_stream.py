"""Keep inline model-only place/source refs out of the public answer stream."""

from __future__ import annotations

import re

_INTERNAL_REF = re.compile(r"\[\[(?:plc|src)_[0-9a-f]{10}\]\]")


class InternalRefStreamFilter:
    """Remove exact ``[[plc_...]]``/``[[src_...]]`` markers across chunks.

    The model emits these markers so the backend can select trusted map pins
    and citations from the complete raw answer. The client must never see them,
    even when a provider splits one marker over several streaming chunks.
    """

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        """Consume a provider chunk and return only text safe for the client."""

        if not chunk:
            return []
        self._pending += chunk
        visible: list[str] = []

        while self._pending:
            match = _INTERNAL_REF.search(self._pending)
            if match is not None:
                if match.start():
                    prefix = self._pending[: match.start()]
                    # Markers are normally separated from prose by one space;
                    # remove that implementation-only separator as well.
                    visible.append(prefix[:-1] if prefix.endswith(" ") else prefix)
                self._pending = self._pending[match.end() :]
                continue

            partial_at = self._partial_ref_start()
            if partial_at is None:
                visible.append(self._pending)
                self._pending = ""
            elif partial_at:
                # Keep a possible separator buffered until we know whether
                # the bracket starts an internal marker or ordinary Markdown.
                buffer_at = partial_at - 1 if self._pending[partial_at - 1] == " " else partial_at
                visible.append(self._pending[:buffer_at])
                self._pending = self._pending[buffer_at:]
            break

        return visible

    def finish(self) -> list[str]:
        """Flush ordinary trailing text and discard a malformed partial marker."""

        if not self._pending:
            return []
        pending = self._pending
        self._pending = ""
        if self._is_partial_ref(pending):
            return []
        return [pending]

    def _partial_ref_start(self) -> int | None:
        """Return the start of a suffix that could become an internal marker."""

        for index in range(len(self._pending)):
            suffix = self._pending[index:]
            if self._is_partial_ref(suffix):
                return index
        return None

    @staticmethod
    def _is_partial_ref(value: str) -> bool:
        if value == "[":
            return True
        if not value.startswith("[["):
            return False

        body = value[2:]
        for prefix in ("plc_", "src_"):
            if prefix.startswith(body):
                return True
            if not body.startswith(prefix):
                continue
            remainder = body[len(prefix) :]
            hex_part = remainder[:10]
            if any(character not in "0123456789abcdef" for character in hex_part):
                return False
            if len(hex_part) < 10:
                return True
            closing = remainder[10:]
            return closing in ("", "]")
        return False
