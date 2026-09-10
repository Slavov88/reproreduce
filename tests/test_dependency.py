from __future__ import annotations

import ast

import pytest

from reproreduce.reduce.dependency_v2 import (
    _control_flow_candidates,
    _expression_candidates,
    _function_candidates,
    apply_candidate as apply_v2_candidate,
)
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


def test_v2_coordinates_unused_parameters_and_calls() -> None:
    source = """
def helper(used, unused):
    return used

value = helper(1, 2)
"""
    candidates = _function_candidates(source)
    assert any(candidate.name == "unused" for candidate in candidates)
    candidate = next(candidate for candidate in candidates if candidate.name == "unused")
    rendered = apply_v2_candidate(source, candidate)
    assert "def helper(used):" in rendered
    assert "helper(1)" in rendered
    ast.parse(rendered)


def test_v2_control_flow_and_expression_candidates_render() -> None:
    source = """
if True:
    value = {'keep': 1, 'drop': 2}
else:
    value = {'other': 3}
raise RuntimeError('target')
"""
    control = _control_flow_candidates(source)
    expressions = _expression_candidates(source)
    assert control and expressions
    for candidate in control + expressions:
        ast.parse(apply_v2_candidate(source, candidate))


def test_v2_oracle_preserves_target_on_small_fixture(tmp_path) -> None:
    from reproreduce.api import reduce
    from reproreduce.oracle.exception import ExceptionOracle

    program = tmp_path / "program.py"
    program.write_text(
        "def helper(used, unused):\n"
        "    return used\n\n"
        "value = helper(1, 2)\n"
        "if value:\n"
        "    raise RuntimeError('V2_TARGET')\n",
        encoding="utf-8",
    )
    result = reduce(
        program,
        oracle=ExceptionOracle(exception_type="RuntimeError", message_regex="V2_TARGET"),
        strategy="dependency_v2",
        timeout=5,
    )
    assert "V2_TARGET" in result.reduced_run.stderr
    assert result.metrics["strategy"] == "dependency_v2"
    assert result.metrics["v2_parameter_call_candidates"] >= 1


def test_invalid_strategy_is_rejected(tmp_path) -> None:
    from reproreduce.api import reduce

    program = tmp_path / "program.py"
    program.write_text("raise RuntimeError('target')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strategy"):
        reduce(program, strategy="unknown")
