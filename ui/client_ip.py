"""Validation helpers for the browser-supplied public client address."""

from __future__ import annotations

import ipaddress


def normalize_public_ip(value: object) -> str | None:
    """Return a canonical globally routable IP address, or ``None``."""

    if not isinstance(value, str):
        return None
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    return str(address) if address.is_global else None
