"""
Test that inequality constraints from PDDL are correctly enforced

Verifies that actions with (not (= ?x ?y)) preconditions
properly reject invalid unifications.
"""

import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent / "src")
if _parent not in sys.path:
    sys.path.insert(0, str(_parent))

from stage3_code_generation.lifted_planner import LiftedPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_inequality_constraints():
    """Test inequality constraint enforcement"""
    print("="*80)
    print("INEQUALITY CONSTRAINT TEST")
    print("="*80)

    # Load blocksworld domain (which has inequality constraints)
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    planner = LiftedPlanner(domain)

    # Check that actions have inequality constraints extracted
    print("\n1. Checking extracted inequality constraints from PDDL:")
    for action in planner._abstract_actions:
        if action.inequality_constraints:
            print(f"   {action.action.name}: {action.inequality_constraints}")

    # Test goal that would fail if constraints not enforced
    # Goal: on(?X, ?Y) - block on another block
    goal_preds = [PredicateAtom("on", ["?X", "?Y"])]

    print("\n2. Testing exploration with goal on(?X, ?Y)")
    print("   This should NOT generate states like on(?X, ?X) (self-loop)")

    state_graph = planner.explore_from_goal(goal_preds, max_states=1000)

    print(f"\n3. Checking generated states for constraint violations:")
    violations = []

    for state in state_graph.states:
        for pred in state.predicates:
            # Check for on(?X, ?X) where same variable appears twice
            if pred.name == "on" and len(pred.args) == 2:
                arg0, arg1 = pred.args
                # Both are variables and same
                if arg0.startswith('?') and arg1.startswith('?') and arg0 == arg1:
                    violations.append((state, pred))
                # Both are concrete and same
                elif not arg0.startswith('?') and not arg1.startswith('?') and arg0 == arg1:
                    violations.append((state, pred))

    if violations:
        print(f"   ❌ FOUND {len(violations)} CONSTRAINT VIOLATIONS:")
        for state, pred in violations[:5]:  # Show first 5
            print(f"      {pred} in state {state}")
        return False
    else:
        print(f"   ✅ NO VIOLATIONS FOUND in {len(state_graph.states)} states")
        print(f"   All states respect inequality constraints from PDDL")
        return True


if __name__ == "__main__":
    success = test_inequality_constraints()
    print("\n" + "="*80)
    if success:
        print("✅ INEQUALITY CONSTRAINT TEST PASSED")
    else:
        print("❌ INEQUALITY CONSTRAINT TEST FAILED")
    print("="*80)
    sys.exit(0 if success else 1)
