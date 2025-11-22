"""
Manual Trace Validation for Backward Search

This test validates the backward search implementation against manual traces
following the user's instructions step by step.
"""

from pathlib import Path
import sys

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_manual_trace_on_a_b():
    """
    Manual trace for goal: on(a, b)

    Following the user's example:

    Goal: on(a, b)

    Step 1: Find actions with +on(?b1, ?b2) in additive effects
    Action: put-on-block(?b1, ?b2)
    - Additive effects: +on(?b1, ?b2), +handempty, +clear(?b1), -holding(?b1), -clear(?b2)
    - Binding: ?b1=a, ?b2=b

    Step 2: Apply regression formula
    goal ∧ prec ∧ deleted_effects ∧ ¬additive_effects
    = on(a,b) ∧ [holding(a) ∧ clear(b)] ∧ [holding(a) ∧ clear(b)] ∧ ¬[on(a,b) ∧ handempty ∧ clear(a)]
    = [holding(a) ∧ clear(b) ∧ holding(a) ∧ clear(b)] - on(a,b) - handempty - clear(a)
    = holding(a) ∧ clear(b)

    Next goal: holding(a) ∧ clear(b)
    """
    print("="*80)
    print("Manual Trace Validation: on(a, b)")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    planner = BackwardSearchPlanner(domain)

    # Test goal
    goal = [PredicateAtom("on", ["a", "b"])]
    print(f"\nInitial Goal: {[str(p) for p in goal]}")

    # Search with depth=1 to see first level
    graph = planner.search(goal, max_states=50, max_depth=1)

    print(f"\nSearch Result (depth=1):")
    print(f"  States: {len(graph.states)}")
    print(f"  Transitions: {len(graph.transitions)}")

    # Find states at depth=1
    depth1_states = [s for s in graph.states if s.depth == 1]
    print(f"\nStates at depth=1: {len(depth1_states)}")
    for i, state in enumerate(depth1_states[:3]):  # Show first 3
        print(f"  {i+1}. {state}")

    # Check if we have holding(a) ∧ clear(b)
    expected_predicates = {
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    }

    found = False
    for state in depth1_states:
        if state.predicates == frozenset(expected_predicates):
            found = True
            print(f"\n✓ FOUND expected state: holding(a) ∧ clear(b)")
            break

    if not found:
        print(f"\n✗ Expected state NOT found")
        print(f"  Expected: holding(a) ∧ clear(b)")

    print("\n" + "="*80)


def test_manual_trace_holding_a():
    """
    Manual trace for goal: holding(a)

    Goal: holding(a) (from previous example)

    Step 1: Find actions with +holding(?b1) in additive effects
    Action: pick-up(?b1, ?b2)
    - Preconditions: handempty, clear(?b1), on(?b1, ?b2), ?b1 ≠ ?b2
    - Additive effects: +holding(?b1), +clear(?b2)
    - Deletion effects: -handempty, -clear(?b1), -on(?b1, ?b2)
    - Binding: ?b1=a, ?b2=? (UNBOUND)

    Step 2: Generate new variable for ?b2
    Parent max_var = 0 (from goal holding(a))
    New variable: ?1
    Complete binding: ?b1=a, ?b2=?1

    Step 3: Apply regression formula
    goal ∧ prec ∧ deleted_effects ∧ ¬additive_effects
    = holding(a) ∧ [handempty ∧ clear(a) ∧ on(a,?1) ∧ a≠?1] ∧ [handempty ∧ clear(a) ∧ on(a,?1)] ∧ ¬[holding(a) ∧ clear(?1)]
    = [handempty ∧ clear(a) ∧ on(a,?1) ∧ a≠?1 ∧ handempty ∧ clear(a) ∧ on(a,?1)] - holding(a) - clear(?1)
    = handempty ∧ clear(a) ∧ on(a,?1) ∧ a≠?1

    Next goal: clear(a) ∧ handempty ∧ on(a,?1) with constraint a≠?1
    """
    print("="*80)
    print("Manual Trace Validation: holding(a)")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    planner = BackwardSearchPlanner(domain)

    # Test goal
    goal = [PredicateAtom("holding", ["a"])]
    print(f"\nInitial Goal: {[str(p) for p in goal]}")

    # Search with depth=1
    graph = planner.search(goal, max_states=50, max_depth=1)

    print(f"\nSearch Result (depth=1):")
    print(f"  States: {len(graph.states)}")
    print(f"  Transitions: {len(graph.transitions)}")

    # Find states at depth=1
    depth1_states = [s for s in graph.states if s.depth == 1]
    print(f"\nStates at depth=1: {len(depth1_states)}")
    for i, state in enumerate(depth1_states[:5]):  # Show first 5
        print(f"  {i+1}. {state}")

    # Check for states with pattern: handempty ∧ clear(a) ∧ on(a, ?X)
    print(f"\nLooking for states matching: handempty ∧ clear(a) ∧ on(a, ?X) ∧ a≠?X")

    found = False
    for state in depth1_states:
        has_handempty = any(p.name == "handempty" for p in state.predicates)
        has_clear_a = any(p.name == "clear" and p.args == ("a",) for p in state.predicates)
        has_on_a = any(p.name == "on" and p.args[0] == "a" and p.args[1].startswith("?") for p in state.predicates)

        if has_handempty and has_clear_a and has_on_a:
            found = True
            print(f"\n✓ FOUND matching state: {state}")
            break

    if not found:
        print(f"\n✗ Expected pattern NOT found")

    print("\n" + "="*80)


def test_conjunction_destruction():
    """
    Test conjunction destruction: holding(a) ∧ clear(b)

    According to user instructions:
    - Process holding(a) first → find actions achieving holding(a)
    - Process clear(b) separately → find actions achieving clear(b)
    - Create separate exploration branches
    """
    print("="*80)
    print("Conjunction Destruction Test: holding(a) ∧ clear(b)")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    planner = BackwardSearchPlanner(domain)

    # Test goal
    goal = [
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    ]
    print(f"\nInitial Goal: {[str(p) for p in goal]}")

    # Search with depth=1
    graph = planner.search(goal, max_states=100, max_depth=1)

    print(f"\nSearch Result (depth=1):")
    print(f"  States: {len(graph.states)}")
    print(f"  Transitions: {len(graph.transitions)}")

    # Find states at depth=1
    depth1_states = [s for s in graph.states if s.depth == 1]
    print(f"\nStates at depth=1: {len(depth1_states)}")

    # Count states that came from processing holding(a)
    states_from_holding = []
    states_from_clear = []

    for state in depth1_states:
        # States from processing holding(a) should have clear(b) but not holding(a)
        has_clear_b = any(p.name == "clear" and "b" in p.args for p in state.predicates)
        has_holding_a = any(p.name == "holding" and "a" in p.args for p in state.predicates)

        if has_clear_b and not has_holding_a:
            states_from_holding.append(state)

        # States from processing clear(b) should have holding(a) but not clear(b)
        if has_holding_a and not has_clear_b:
            states_from_clear.append(state)

    print(f"\nStates from processing holding(a) branch: {len(states_from_holding)}")
    for i, state in enumerate(states_from_holding[:3]):
        print(f"  {i+1}. {state}")

    print(f"\nStates from processing clear(b) branch: {len(states_from_clear)}")
    for i, state in enumerate(states_from_clear[:3]):
        print(f"  {i+1}. {state}")

    if len(states_from_holding) > 0 and len(states_from_clear) > 0:
        print(f"\n✓ PASS: Conjunction destruction working (both branches explored)")
    else:
        print(f"\n✗ FAIL: Conjunction destruction not working properly")

    print("\n" + "="*80)


def run_all_tests():
    """Run all manual trace validation tests"""
    print("\n" + "="*80)
    print("BACKWARD SEARCH MANUAL TRACE VALIDATION")
    print("="*80 + "\n")

    test_manual_trace_on_a_b()
    test_manual_trace_holding_a()
    test_conjunction_destruction()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
