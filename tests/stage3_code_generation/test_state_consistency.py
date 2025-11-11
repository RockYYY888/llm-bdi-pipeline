"""
Test State Consistency Validation

Verifies that the forward planner only generates physically valid states
by checking that state consistency validation works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.state_space import PredicateAtom
from src.utils.pddl_parser import PDDLParser


def check_state_validity(state, verbose=False):
    """
    Check if a state is physically valid

    Returns (is_valid, violations) where violations is a list of issues
    """
    violations = []
    predicates = list(state.predicates)

    # Extract predicates by type
    handempty = any(p.name == 'handempty' for p in predicates)
    holding = [p for p in predicates if p.name == 'holding']
    ontable = [p for p in predicates if p.name == 'ontable']
    on = [p for p in predicates if p.name == 'on']
    clear = [p for p in predicates if p.name == 'clear']

    # Check 1: Hand contradictions
    if handempty and len(holding) > 0:
        violations.append("hand contradiction: both handempty and holding")

    # Check 2: Multiple holdings
    if len(holding) > 1:
        violations.append(f"multiple holdings: {[str(p) for p in holding]}")

    # Check 3: Circular on-relationships
    on_map = {}
    for pred in on:
        if len(pred.args) == 2:
            block, base = pred.args
            on_map[block] = base

    for block, base in on_map.items():
        if base in on_map and on_map[base] == block:
            violations.append(f"circular on: on({block},{base}) and on({base},{block})")

        # Check for indirect cycles
        visited = set()
        current = base
        while current in on_map:
            if current in visited:
                violations.append(f"cycle detected involving {block}")
                break
            visited.add(current)
            current = on_map[current]
            if current == block:
                violations.append(f"cycle: {block} -> ... -> {block}")
                break

    # Check 4: Location contradictions
    ontable_blocks = {pred.args[0] for pred in ontable if len(pred.args) == 1}
    on_blocks = {pred.args[0] for pred in on if len(pred.args) == 2}

    if ontable_blocks & on_blocks:
        common = ontable_blocks & on_blocks
        violations.append(f"location contradiction: {common} both ontable and on another block")

    # Check 5: Clear contradictions
    clear_blocks = {pred.args[0] for pred in clear if len(pred.args) == 1}
    base_blocks = {pred.args[1] for pred in on if len(pred.args) == 2}

    if clear_blocks & base_blocks:
        common = clear_blocks & base_blocks
        violations.append(f"clear contradiction: {common} marked clear but has block on top")

    if verbose and violations:
        print(f"\nInvalid state: {sorted([str(p) for p in predicates])}")
        for v in violations:
            print(f"  - {v}")

    return len(violations) == 0, violations


def test_2_blocks():
    """Test with 2 blocks"""
    print("="*80)
    print("STATE CONSISTENCY TEST: 2 Blocks")
    print("="*80)

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Test goal: on(a, b)
    goal_preds = [PredicateAtom('on', ('a', 'b'))]
    objects = ['a', 'b']

    planner = ForwardStatePlanner(domain, objects)
    graph = planner.explore_from_goal(goal_preds)

    print(f"\nExplored {len(graph.states)} states")

    # Check all states
    invalid_count = 0
    invalid_examples = []

    for state in graph.states:
        is_valid, violations = check_state_validity(state)
        if not is_valid:
            invalid_count += 1
            if len(invalid_examples) < 5:  # Keep first 5 examples
                invalid_examples.append((state, violations))

    if invalid_count == 0:
        print(f"✅ PASS: All {len(graph.states)} states are valid!")
        return True
    else:
        print(f"❌ FAIL: Found {invalid_count} invalid states!")
        print(f"\nFirst {len(invalid_examples)} examples:")
        for i, (state, violations) in enumerate(invalid_examples, 1):
            print(f"\n{i}. {sorted([str(p) for p in state.predicates])}")
            for v in violations:
                print(f"   - {v}")
        return False


def test_3_blocks():
    """Test with 3 blocks"""
    print("\n\n" + "="*80)
    print("STATE CONSISTENCY TEST: 3 Blocks")
    print("="*80)

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Test goal: on(a, b)
    goal_preds = [PredicateAtom('on', ('a', 'b'))]
    objects = ['a', 'b', 'c']

    planner = ForwardStatePlanner(domain, objects)
    graph = planner.explore_from_goal(goal_preds)

    print(f"\nExplored {len(graph.states)} states")

    # Check all states
    invalid_count = 0
    invalid_examples = []

    for state in graph.states:
        is_valid, violations = check_state_validity(state)
        if not is_valid:
            invalid_count += 1
            if len(invalid_examples) < 5:  # Keep first 5 examples
                invalid_examples.append((state, violations))

    if invalid_count == 0:
        print(f"✅ PASS: All {len(graph.states)} states are valid!")
        return True
    else:
        print(f"❌ FAIL: Found {invalid_count} invalid states!")
        print(f"\nFirst {len(invalid_examples)} examples:")
        for i, (state, violations) in enumerate(invalid_examples, 1):
            print(f"\n{i}. {sorted([str(p) for p in state.predicates])}")
            for v in violations:
                print(f"   - {v}")
        return False


if __name__ == "__main__":
    print("Starting State Consistency Validation Tests")
    print()

    # Test with 2 blocks
    try:
        success1 = test_2_blocks()
    except Exception as e:
        print(f"\n❌ 2 blocks test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success1 = False

    # Test with 3 blocks
    try:
        success2 = test_3_blocks()
    except Exception as e:
        print(f"\n❌ 3 blocks test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success2 = False

    # Summary
    print("\n\n" + "="*80)
    print("STATE CONSISTENCY TEST SUMMARY")
    print("="*80)
    print(f"2 Blocks Test:    {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"3 Blocks Test:    {'✅ PASS' if success2 else '❌ FAIL'}")
    print()

    if success1 and success2:
        print("✅ ALL STATE CONSISTENCY TESTS PASSED")
        print("\nNo invalid states were generated!")
        print("The state consistency validation is working correctly.")
        exit(0)
    else:
        print("❌ SOME STATE CONSISTENCY TESTS FAILED")
        exit(1)
