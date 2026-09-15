"""Architecture tests for one-way dependencies inside ``tools.geo``."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

_GEO_ROOT = Path(__file__).parents[1] / "geo"


def _imports_below(scope: str) -> Iterator[tuple[Path, str]]:
    root = _GEO_ROOT / scope
    paths = root.rglob("*.py") if root.is_dir() else (root,)
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                yield path, node.module
            elif isinstance(node, ast.Import):
                yield from ((path, alias.name) for alias in node.names)


@pytest.mark.parametrize(
    ("scope", "forbidden_prefixes"),
    [
        (
            "geocoding",
            (
                "tools.geo.places_search",
                "tools.geo.routing",
            ),
        ),
    ],
)
def test_geo_package_dependencies_point_one_way(
    scope: str,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    violations = [
        f"{path.relative_to(_GEO_ROOT)} imports {module}"
        for path, module in _imports_below(scope)
        if module.startswith(forbidden_prefixes)
    ]

    assert violations == []
