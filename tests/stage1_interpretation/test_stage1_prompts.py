import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.prompts import get_ltl_system_prompt, get_ltl_user_prompt


def test_stage1_system_prompt_retains_full_ltlf_operator_coverage():
    prompt = get_ltl_system_prompt(
        "demo",
        "items",
        "- linked(?x, ?y)\n- ready(?x)\n- available",
        "- act(?x)",
        "- do_link(?x, ?y)",
    )

    assert "Task-grounded query rule:" in prompt
    assert "LTLf SYNTAX REFERENCE:" in prompt
    assert "SUPPORTED OPERATOR PRECEDENCE" in prompt
    assert '"type": "release"' in prompt
    assert '"type": "until"' in prompt
    assert '"type": "nested"' in prompt
    assert '"operator": "X"' in prompt
    assert '"operator": "WX"' in prompt
    assert '"operator": "F"' in prompt
    assert '"operator": "G"' in prompt
    assert '"type": "implication"' in prompt
    assert '"type": "equivalence"' in prompt
    assert '"type": "negation"' in prompt
    assert "HIGH-VALUE FEW-SHOT PATTERNS:" in prompt
    assert '"operator": "WX"' in prompt
    assert '"type": "release"' in prompt


def test_stage1_system_prompt_preserves_schema_and_boundary_rules():
    prompt = get_ltl_system_prompt(
        "demo",
        "items",
        "- linked(?x, ?y)\n- ready(?x)\n- available",
    )

    assert 'Return exactly one JSON object with keys "objects", "ltl_formulas", and "atoms".' in prompt
    assert "Do not assume any hidden initial state, benchmark metadata, problem.hddl facts" in prompt
    assert "Never emit task names as LTL atoms." in prompt
    assert "Every predicate instance used in ltl_formulas must have a corresponding atoms entry." in prompt
    assert "Examples below are schematic." in prompt
    assert "Do not add unstated support predicates" in prompt
    assert "complete the tasks A(...), B(...), and C(...)" in prompt
    assert "prefer a conjunction of independent eventual goals" in prompt
    assert '"A": ["a"]' in prompt
    assert 'do not collapse this into one {"type": "temporal", "operator": "F"' in prompt


def test_stage1_system_prompt_keeps_high_risk_operator_disambiguation_examples():
    prompt = get_ltl_system_prompt(
        "demo",
        "items",
        "- on(?x, ?y)\n- clear(?x)\n- holding(?x)\n- ontable(?x)",
    )

    assert 'Pick up a then immediately place on b' in prompt
    assert 'In next state if exists a is on b' in prompt
    assert 'b stays clear unless a is on table' in prompt


def test_stage1_user_prompt_keeps_goal_prefix():
    assert get_ltl_user_prompt("Do something.") == "Goal: Do something."
