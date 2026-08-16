# tests/test_python_floor.py
"""The declared Python floor must be the floor CI actually exercises.

`requires-python` is a promise to downstream consumers (issue #95: a consumer
supporting 3.11 wants to pin gpudge). The promise is only worth something if
CI runs the suite on that version, so this guards the two declarations against
drifting apart: raising the pyproject floor without dropping the matrix cell,
or dropping the matrix cell without raising the floor, both fail here.

Everything takes text rather than reading files, so the guard's own behaviour is
testable against fixtures with no need to mutate the real pyproject.toml.
"""
import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

_CELL = re.compile(r"\d+\.\d+")


def parse_floor(pyproject_text: str) -> tuple[int, int]:
    """`(major, minor)` from a ``requires-python = ">=X.Y"`` declaration."""
    spec = tomllib.loads(pyproject_text)["project"]["requires-python"]
    m = re.fullmatch(r">=\s*(\d+)\.(\d+)", spec.strip())
    if m is None:
        raise ValueError(f"expected a `>=X.Y` requires-python spec, got {spec!r}")
    return int(m.group(1)), int(m.group(2))


def parse_matrix(ci_text: str) -> list[tuple[int, int]]:
    """Sorted `(major, minor)` cells of the CI ``pytest`` job's Python matrix.

    Cells must be QUOTED strings: `yaml.safe_load` reads an unquoted `3.10` as
    the float `3.1`, which would silently parse as Python 3.1.
    """
    workflow = yaml.safe_load(ci_text)
    cells = workflow["jobs"]["pytest"]["strategy"]["matrix"]["python-version"]
    parsed = []
    for cell in cells:
        if not isinstance(cell, str) or not _CELL.fullmatch(cell):
            raise ValueError(
                f"CI matrix cell {cell!r} must be a quoted 'X.Y' string; YAML "
                "reads an unquoted 3.10 as the float 3.1."
            )
        major, minor = cell.split(".")
        parsed.append((int(major), int(minor)))
    return sorted(parsed)


def validate_floor_matches(pyproject_text: str, ci_text: str) -> tuple[int, int]:
    """Return the agreed floor, or raise AssertionError describing the drift."""
    floor = parse_floor(pyproject_text)
    cells = parse_matrix(ci_text)
    if not cells:
        raise AssertionError("the CI Python matrix is empty")
    if cells[0] != floor:
        raise AssertionError(
            f"CI's lowest Python {cells[0]} is not the declared requires-python "
            f"floor {floor} — pyproject.toml and .github/workflows/ci.yml were "
            "changed independently."
        )
    return floor


# --- the live declarations ------------------------------------------------

def test_declared_floor_is_311():
    assert parse_floor(PYPROJECT.read_text()) == (3, 11)


def test_ci_matrix_lowest_cell_is_the_declared_floor():
    assert validate_floor_matches(PYPROJECT.read_text(), CI.read_text()) == (3, 11)


# --- the guard's own guard: exercise validate_floor_matches on fixtures ---

_PYPROJECT_311 = '[project]\nname = "gpudge"\nrequires-python = ">=3.11"\n'
_PYPROJECT_312 = '[project]\nname = "gpudge"\nrequires-python = ">=3.12"\n'
_CI_BOTH = (
    "jobs:\n"
    "  pytest:\n"
    "    strategy:\n"
    "      matrix:\n"
    '        python-version: ["3.11", "3.12"]\n'
)
_CI_312_ONLY = (
    "jobs:\n"
    "  pytest:\n"
    "    strategy:\n"
    "      matrix:\n"
    '        python-version: ["3.12"]\n'
)


def test_validate_accepts_matching_declarations():
    assert validate_floor_matches(_PYPROJECT_311, _CI_BOTH) == (3, 11)


def test_validate_rejects_floor_raised_while_matrix_still_covers_311():
    with pytest.raises(AssertionError, match="changed independently"):
        validate_floor_matches(_PYPROJECT_312, _CI_BOTH)


def test_validate_rejects_matrix_cell_dropped_while_floor_stays_311():
    with pytest.raises(AssertionError, match="changed independently"):
        validate_floor_matches(_PYPROJECT_311, _CI_312_ONLY)


# --- parser contracts ------------------------------------------------------

def test_matrix_parser_accepts_block_sequence_syntax():
    block = (
        "jobs:\n"
        "  pytest:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version:\n"
        "          - '3.11'\n"
        "          - '3.12'\n"
    )
    assert parse_matrix(block) == [(3, 11), (3, 12)]


def test_matrix_parser_rejects_unquoted_version_cells():
    unquoted = (
        "jobs:\n"
        "  pytest:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: [3.10, 3.12]\n"
    )
    with pytest.raises(ValueError, match="quoted"):
        parse_matrix(unquoted)


def test_floor_parser_rejects_a_spec_it_cannot_reason_about():
    with pytest.raises(ValueError):
        parse_floor('[project]\nrequires-python = ">=3.11,<4"\n')
