"""
Test mutex validation in lifted planning

This test verifies that the mutex validation correctly filters out invalid states
with conflicting predicates like {handempty, holding(?X)}.
"""

import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent / "src")
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from stage3_code_generation.lifted_planner import LiftedPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser


def check_for_invalid_states(states):
    """
    Check for common invalid state patterns

    Returns:
        Dict with counts of different types of invalid states
    """
    invalid_counts = {
        'handempty_and_holding': 0,
        'multiple_holding': 0,
        'self_loop_on': 0,
        'duplicate_single_arg': 0
    }

    invalid_examples = {
        'handempty_and_holding': [],
        'multiple_holding': [],
        'self_loop_on': [],
        'duplicate_single_arg': []
    }

    for state in states:
        pred_names = {}
        for pred in state.predicates:
            if pred.name not in pred_names:
                pred_names[pred.name] = []
            pred_names[pred.name].append(pred)

        # Check 1: handempty and holding together (mutex)
        if 'handempty' in pred_names and 'holding' in pred_names:
            invalid_counts['handempty_and_holding'] += 1
            if len(invalid_examples['handempty_and_holding']) < 3:
                invalid_examples['handempty_and_holding'].append(str(state))

        # Check 2: Multiple holding predicates
        if 'holding' in pred_names and len(pred_names['holding']) > 1:
            invalid_counts['multiple_holding'] += 1
            if len(invalid_examples['multiple_holding']) < 3:
                invalid_examples['multiple_holding'].append(str(state))

        # Check 3: Self-loop in on(X, X)
        if 'on' in pred_names:
            for on_pred in pred_names['on']:
                if len(on_pred.args) == 2:
                    arg0, arg1 = on_pred.args
                    # Only check concrete values (not variables)
                    if not arg0.startswith('?') and not arg1.startswith('?') and arg0 == arg1:
                        invalid_counts['self_loop_on'] += 1
                        if len(invalid_examples['self_loop_on']) < 3:
                            invalid_examples['self_loop_on'].append(str(state))
                        break

        # Check 4: Duplicate single-argument predicates with different concrete values
        for pred_name, preds in pred_names.items():
            if len(preds) > 1:
                concrete_args = []
                for p in preds:
                    if len(p.args) == 1 and not p.args[0].startswith('?'):
                        concrete_args.append(p.args[0])
                if len(set(concrete_args)) > 1:
                    invalid_counts['duplicate_single_arg'] += 1
                    if len(invalid_examples['duplicate_single_arg']) < 3:
                        invalid_examples['duplicate_single_arg'].append(str(state))
                    break

    return invalid_counts, invalid_examples


def test_mutex_validation():
    """
    Test that mutex validation correctly filters invalid states
    """
    print("="*80)
    print("MUTEX VALIDATION TEST")
    print("="*80)

    # Load domain
    domain_file = Path(__file__).parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Domain: {domain.name}\n")

    # Test with holding(?X) goal
    goal_preds = [PredicateAtom("holding", ["?X"])]

    planner = LiftedPlanner(domain)
    print(f"Goal: {[str(p) for p in goal_preds]}")
    print(f"Extracted mutex predicates: {planner._mutex_predicates}")
    print("\nStarting exploration...\n")

    result = planner.explore_from_goal(goal_preds, max_states=1000)

    print(f"\nExploration Result:")
    print(f"  States: {len(result['states']):,}")
    print(f"  Transitions: {len(result['transitions']):,}\n")

    # Check for invalid states
    invalid_counts, invalid_examples = check_for_invalid_states(result['states'])

    print("="*80)
    print("INVALID STATE ANALYSIS")
    print("="*80)

    total_invalid = sum(invalid_counts.values())

    if total_invalid == 0:
        print("✅ NO INVALID STATES FOUND!")
        print("\nMutex validation is working correctly.")
    else:
        print(f"❌ FOUND {total_invalid} INVALID STATES:\n")

        for check_type, count in invalid_counts.items():
            if count > 0:
                print(f"  {check_type}: {count} states")
                print(f"  Examples:")
                for example in invalid_examples[check_type]:
                    print(f"    - {example}")
                print()

        print("⚠️  Mutex validation may not be working correctly!")

    print("="*80)

    return total_invalid == 0


if __name__ == "__main__":
    success = test_mutex_validation()
    sys.exit(0 if success else 1)
