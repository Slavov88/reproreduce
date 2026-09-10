from __future__ import annotations

import ast

import pytest

from reproreduce.reduce.dependency import (
    analyze_source,
    apply_candidate,
    dependency_candidates,
    expression_candidates,
    import_alias_candidates,
)


def test_dependency_analysis_tracks_bindings_calls_and_dynamic_features() -> None:
    source = """
import math, json

def live(value):
    return math.sin(value)

def dead():
    return 1

def dead_again():
    return 2

result = live(1.0)
"""
    analysis = analyze_source(source)
    kinds = {(item.kind, next(iter(item.defines), "")) for item in analysis.statements}
    assert ("FunctionDef", "live") in kinds
    assert ("FunctionDef", "dead") in kinds
    assert analysis.dynamic_features is False
    assert analysis.component_count > 0
    candidates = dependency_candidates(analysis)
    assert any(candidate.kind == "unused_definition" for candidate in candidates)
    assert any(candidate.kind == "dependency_component" for candidate in candidates)


def test_dynamic_python_is_marked_for_conservative_reporting() -> None:
    analysis = analyze_source("value = eval('1')\n")
    assert analysis.dynamic_features is True


def test_import_alias_and_expression_candidates_render_valid_python() -> None:
    source = """
import math, json
value = math.sin(1.0) + 2.0
message = 'x' if value > 0 else 'y'
"""
    candidates = import_alias_candidates(source) + expression_candidates(source)
    assert candidates
    for candidate in candidates:
        rendered = apply_candidate(source, candidate)
        ast.parse(rendered)


def test_no_unused_candidate_for_referenced_definition() -> None:
    analysis = analyze_source("def live():\n    return 1\nvalue = live()\n")
    assert not any(
        candidate.kind == "unused_definition" and "live" in candidate.label
        for candidate in dependency_candidates(analysis)
    )


def test_invalid_strategy_is_rejected(tmp_path) -> None:
    from reproreduce.api import reduce

    program = tmp_path / "program.py"
    program.write_text("raise RuntimeError('target')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strategy"):
        reduce(program, strategy="unknown")
