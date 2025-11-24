"""
Test backward search implementation against detailed instruction

This test verifies that the backward search implementation correctly follows
the instruction provided, including:
1. Variable numbering (?1, ?2, ?3 format)
2. Regression formula: goal ∧ prec ∧ del_effects ∧ ¬add_effects
3. Conjunction destruction (one predicate at a time)
4. Inequality constraints support
5. Partial variable bindings (e.g., pick-up(a, ?1))
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def test_1_simple_goal_on_a_b():
    """
    Test 1: Simple goal on(a, b)

    Expected behavior following instruction:
    1. Goal: on(a, b)
    2. Find action with +on(a, b): put-on-block(?b1, ?b2)
    3. Bind ?b1=a, ?b2=b
    4. Apply regression:
       - Remove additive effects: on(a,b), handempty, clear(a)
       - Add preconditions: holding(a), clear(b)
       - Add deletion effects: holding(a), clear(b)
       Result: holding(a) ∧ clear(b)
    """
    print("=" * 80)
    print("TEST 1: Simple goal on(a, b)")
    print("=" * 80)

    # Load domain
    domain_file = src_path / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Goal: on(a, b)
    goal = [PredicateAtom("on", ["a", "b"])]

    print(f"\nGoal: {[str(p) for p in goal]}")
    print("\nExpected first expansion:")
    print("  Action: put-on-block(a, b)")
    print("  Regression result: holding(a) ∧ clear(b)")
    print("\nActual exploration:")

    # Search
    state_graph = planner.search(goal, max_states=100)

    print(f"\nResults:")
    print(f"  States explored: {len(state_graph.states)}")
    print(f"  Transitions: {len(state_graph.transitions)}")

    # Verify: Should have created a state with holding(a) and clear(b)
    found_expected_state = False
    for state in state_graph.states:
        pred_strs = {str(p) for p in state.predicates}
        if "holding(a)" in pred_strs and "clear(b)" in pred_strs:
            found_expected_state = True
            print(f"\n✓ Found expected state: {state}")
            break

    if not found_expected_state:
        print("\n✗ ERROR: Expected state 'holding(a) ∧ clear(b)' not found!")

    print("\n" + "=" * 80 + "\n")


def test_2_conjunction_goal():
    """
    Test 2: Conjunction goal holding(a) ∧ clear(b)

    Expected behavior following instruction:
    1. Start with goal: holding(a) ∧ clear(b)
    2. Process holding(a) first:
       - Find action: pick-up(?b1, ?b2)
       - Bind: ?b1=a, ?b2=?1 (partial binding!)
       - Regression: clear(b) ∧ handempty ∧ clear(a) ∧ on(a, ?1) ∧ a≠?1
    3. Process clear(b):
       - Find action: pick-up(?b1, ?b2)
       - Bind: ?b2=b, ?b1=?1 (partial binding!)
       - Regression: holding(a) ∧ handempty ∧ clear(?1) ∧ on(?1, b) ∧ ?1≠b
    """
    print("=" * 80)
    print("TEST 2: Conjunction goal holding(a) ∧ clear(b)")
    print("=" * 80)

    # Load domain
    domain_file = src_path / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Goal: holding(a) ∧ clear(b)
    goal = [
        PredicateAtom("holding", ["a"]),
        PredicateAtom("clear", ["b"])
    ]

    print(f"\nGoal: {[str(p) for p in goal]}")
    print("\nExpected expansions:")
    print("  Branch 1 (achieving holding(a)):")
    print("    Action: pick-up(a, ?1)")
    print("    Regression: clear(b) ∧ handempty ∧ clear(a) ∧ on(a, ?1) ∧ (a ≠ ?1)")
    print("  Branch 2 (achieving clear(b)):")
    print("    Action: pick-up(?1, b)")
    print("    Regression: holding(a) ∧ handempty ∧ clear(?1) ∧ on(?1, b) ∧ (?1 ≠ b)")
    print("\nActual exploration:")

    # Search
    state_graph = planner.search(goal, max_states=100)

    print(f"\nResults:")
    print(f"  States explored: {len(state_graph.states)}")
    print(f"  Transitions: {len(state_graph.transitions)}")

    # Verify: Should have states with variables ?1
    found_variable_state = False
    for state in state_graph.states:
        pred_strs = {str(p) for p in state.predicates}
        for pred_str in pred_strs:
            if "?1" in pred_str:
                found_variable_state = True
                print(f"\n✓ Found state with variable ?1: {state}")
                break
        if found_variable_state:
            break

    if not found_variable_state:
        print("\n✗ WARNING: No state with variable ?1 found (expected partial binding)")

    print("\n" + "=" * 80 + "\n")


def test_3_variable_numbering():
    """
    Test 3: Variable numbering verification

    Expected behavior:
    - Variables should be numbered ?1, ?2, ?3, ... (NOT ?v0, ?v1, ?v2)
    - Each new variable increments from parent state's max_var_number
    """
    print("=" * 80)
    print("TEST 3: Variable numbering (?1, ?2, ?3 format)")
    print("=" * 80)

    # Load domain
    domain_file = src_path / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Goal with variable: on(?1, ?2)
    goal = [PredicateAtom("on", ["?1", "?2"])]

    print(f"\nGoal: {[str(p) for p in goal]}")
    print("\nExpected variable format: ?1, ?2, ?3, ... (NOT ?v0, ?v1, ?v2)")
    print("\nActual exploration:")

    # Search
    state_graph = planner.search(goal, max_states=100, max_objects=3)

    print(f"\nResults:")
    print(f"  States explored: {len(state_graph.states)}")

    # Verify: Check all variables use correct format
    all_variables = set()
    invalid_variables = set()

    for state in state_graph.states:
        for pred in state.predicates:
            for arg in pred.args:
                if arg.startswith('?'):
                    all_variables.add(arg)
                    # Check if it's in wrong format (?v0, ?v1, etc.)
                    if arg.startswith('?v') and arg[2:].isdigit():
                        invalid_variables.add(arg)

    print(f"\nAll variables found: {sorted(all_variables)}")

    if invalid_variables:
        print(f"\n✗ ERROR: Found invalid variable format: {invalid_variables}")
        print("  Should use ?1, ?2, ?3, ... NOT ?v0, ?v1, ?v2")
    else:
        print(f"\n✓ All variables use correct format (?1, ?2, ?3, ...)")

    print("\n" + "=" * 80 + "\n")


def test_4_inequality_constraints():
    """
    Test 4: Inequality constraints support

    Expected behavior:
    - pick-up action has precondition: (not (= ?b1 ?b2))
    - This should be extracted and added as inequality constraint
    """
    print("=" * 80)
    print("TEST 4: Inequality constraints (not (= ?b1 ?b2))")
    print("=" * 80)

    # Load domain
    domain_file = src_path / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Check pick-up action parsing
    pick_up_action = None
    for parsed_action in planner.parsed_actions:
        if parsed_action.action.name == "pick-up":
            pick_up_action = parsed_action
            break

    if pick_up_action:
        print(f"\nAction: {pick_up_action.action.name}")
        print(f"Parameters: {pick_up_action.parameters}")
        print(f"Inequality constraints: {pick_up_action.inequality_constraints}")

        if pick_up_action.inequality_constraints:
            print(f"\n✓ Inequality constraints extracted correctly")
        else:
            print(f"\n✗ ERROR: No inequality constraints found for pick-up action")
    else:
        print(f"\n✗ ERROR: pick-up action not found")

    # Now test that constraints propagate to states
    goal = [PredicateAtom("holding", ["a"])]
    print(f"\n\nGoal: {[str(p) for p in goal]}")
    print("\nExpected: States should have inequality constraints")

    state_graph = planner.search(goal, max_states=50)

    # Check if any state has constraints
    found_constraint = False
    for state in state_graph.states:
        if hasattr(state, 'constraints') and state.constraints:
            found_constraint = True
            print(f"\n✓ Found state with constraint: {state}")
            print(f"  Constraints: {state.constraints}")
            break

    if not found_constraint:
        print(f"\n✗ WARNING: No states with inequality constraints found")

    print("\n" + "=" * 80 + "\n")


def test_5_regression_formula_verification():
    """
    Test 5: Verify regression formula is correctly applied

    Formula: goal ∧ prec ∧ deleted_effects ∧ ¬additive_effects

    Example:
    - Current goal: on(a, b)
    - Action: put-on-block(?b1, ?b2)
      - prec: holding(?b1) ∧ clear(?b2)
      - del: holding(?b1), clear(?b2)
      - add: on(?b1, ?b2), handempty, clear(?b1)
    - Binding: ?b1=a, ?b2=b

    Regression:
    1. Start: {on(a,b)}
    2. Remove add effects in goal: {} (on(a,b) removed)
    3. Add prec: {holding(a), clear(b)}
    4. Add del: {holding(a), clear(b)} (already there)
    Result: {holding(a), clear(b)}
    """
    print("=" * 80)
    print("TEST 5: Regression formula verification")
    print("=" * 80)

    from stage3_code_generation.backward_search_refactored import BackwardState

    # Load domain
    domain_file = src_path / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_file))

    # Create planner
    planner = BackwardSearchPlanner(domain)

    # Manual test of regression
    print("\nManual regression test:")
    print("  Current goal: on(a, b)")
    print("  Action: put-on-block(?b1, ?b2)")
    print("    Preconditions: holding(?b1) ∧ clear(?b2)")
    print("    Deletion effects: holding(?b1), clear(?b2)")
    print("    Additive effects: on(?b1, ?b2), handempty, clear(?b1)")
    print("  Binding: ?b1=a, ?b2=b")
    print("\nExpected regression result:")
    print("  1. Start with: {on(a, b)}")
    print("  2. Remove additive effects in goal: on(a,b) removed → {}")
    print("  3. Add preconditions: {holding(a), clear(b)}")
    print("  4. Add deletion effects: {holding(a), clear(b)} (no change)")
    print("  Final: {holding(a), clear(b)}")

    # Create initial state
    current_state = BackwardState(
        predicates={PredicateAtom("on", ["a", "b"])},
        constraints=set(),
        depth=0,
        max_var_number=0
    )

    # Find put-on-block action
    put_on_block = None
    for parsed_action in planner.parsed_actions:
        if parsed_action.action.name == "put-on-block":
            put_on_block = parsed_action
            break

    if put_on_block:
        # Apply regression manually
        target_pred = PredicateAtom("on", ["a", "b"])
        binding = {"?b1": "a", "?b2": "b"}

        predecessors = planner._apply_regression(
            current_state,
            target_pred,
            put_on_block,
            binding
        )

        print("\nActual regression result:")
        for pred_state in predecessors:
            print(f"  {pred_state}")
            pred_strs = {str(p) for p in pred_state.predicates}

            # Verify expected predicates
            if "holding(a)" in pred_strs and "clear(b)" in pred_strs:
                print(f"\n✓ Regression formula correctly applied!")
            else:
                print(f"\n✗ ERROR: Expected {{holding(a), clear(b)}}, got {pred_strs}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  BACKWARD SEARCH INSTRUCTION COMPLIANCE TEST SUITE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run all tests
    test_1_simple_goal_on_a_b()
    test_2_conjunction_goal()
    test_3_variable_numbering()
    test_4_inequality_constraints()
    test_5_regression_formula_verification()

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "  ALL TESTS COMPLETE".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
