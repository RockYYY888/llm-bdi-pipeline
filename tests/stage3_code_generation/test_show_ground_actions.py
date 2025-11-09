"""
Show All 32 Ground Actions for 2 Blocks

This program lists all ground actions generated for 2 blocks {a, b}
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.utils.pddl_parser import PDDLParser
from collections import defaultdict


def show_ground_actions():
    print("="*80)
    print("GROUND ACTIONS FOR 2 BLOCKS {a, b}")
    print("="*80)
    print()

    # Load domain
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # 2 blocks
    objects = ['a', 'b']

    print(f"Objects: {objects}")
    print(f"Domain: blocksworld")
    print()

    # Create planner to access ground actions
    planner = ForwardStatePlanner(domain, objects)

    # Get all ground actions
    all_ground_actions = list(planner._ground_all_actions())

    print(f"Total ground actions: {len(all_ground_actions)}")
    print()

    # Group by action type
    actions_by_type = defaultdict(list)
    for ga in all_ground_actions:
        actions_by_type[ga.action.name].append(ga)

    # Show each action type
    print("="*80)
    print("BREAKDOWN BY ACTION TYPE")
    print("="*80)
    print()

    for action_name in sorted(actions_by_type.keys()):
        ground_actions = actions_by_type[action_name]
        print(f"{action_name}: {len(ground_actions)} groundings")
        print("-" * 40)

        for i, ga in enumerate(ground_actions, 1):
            args_str = ", ".join(ga.args)
            print(f"  {i}. {action_name}({args_str})")

            # Show parameters from action schema
            if i == 1:  # Only show once per action type
                params = ga.action.parameters
                print(f"     Schema: {action_name}({params})")

        print()

    # Explain why 32
    print("="*80)
    print("WHY 32 GROUND ACTIONS?")
    print("="*80)
    print()

    for action_name, ground_actions in sorted(actions_by_type.items()):
        action = ground_actions[0].action
        param_count = len(action.parameters)

        print(f"{action_name}:")
        print(f"  Parameters: {action.parameters}")
        print(f"  Parameter count: {param_count}")
        print(f"  Objects: {objects} (2 objects)")
        print(f"  Possible groundings: 2^{param_count} = {2**param_count}")
        print(f"  Actual groundings: {len(ground_actions)}")

        if len(ground_actions) != 2**param_count:
            print(f"  Note: May have constraints reducing valid groundings")

        print()

    # Calculate total
    total_expected = sum(2**len(ga.action.parameters) for ga in all_ground_actions[:7])  # 7 action types
    print(f"Expected total (sum of all): {len(all_ground_actions)}")
    print()

    # Show action schemas from PDDL
    print("="*80)
    print("ORIGINAL PDDL ACTION SCHEMAS")
    print("="*80)
    print()

    for action in domain.actions:
        print(f"(:action {action.name}")
        print(f"  :parameters ({action.parameters})")
        print(f"  :precondition {action.preconditions[:80]}...")
        print(f"  :effect {action.effects[:80]}...)")
        print()

    # Key insight
    print("="*80)
    print("KEY INSIGHT")
    print("="*80)
    print()

    print("For 2 blocks {a, b}, each action parameter can be bound to:")
    print("  - a")
    print("  - b")
    print()
    print("Examples:")
    print("  pick-up(?x) → 2 groundings: pick-up(a), pick-up(b)")
    print("  put-on-block(?x, ?y) → 4 groundings: put-on-block(a,a), put-on-block(a,b), ...")
    print("  pick-tower(?x, ?y, ?z) → 8 groundings: pick-tower(a,a,a), pick-tower(a,a,b), ...")
    print()
    print("Even 'invalid' combinations are generated:")
    print("  - pick-up(a) when a is not on table (precondition fails)")
    print("  - put-on-block(a, a) (putting block on itself!)")
    print()
    print("These are FILTERED later by precondition checking.")
    print("But they're still GENERATED, contributing to the 32 total.")
    print()


if __name__ == "__main__":
    show_ground_actions()
