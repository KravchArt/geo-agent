from __future__ import annotations

import pytest

from ui.client_ip import normalize_public_ip


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8.8.8.8", "8.8.8.8"),
        (" 2606:4700:4700::1111 ", "2606:4700:4700::1111"),
        ("127.0.0.1", None),
        ("172.24.0.1", None),
        ("not-an-ip", None),
        (None, None),
    ],
)
def test_normalize_public_ip(value: object, expected: str | None) -> None:
    assert normalize_public_ip(value) == expected
