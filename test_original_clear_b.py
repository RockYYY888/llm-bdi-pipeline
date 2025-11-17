"""Test original lifted planning with clear(b) goal"""
import sys
from pathlib import Path

_parent = str(Path(__file__).parent / "src")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.lifted_planner import LiftedPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

domain_file = Path(__file__).parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))

goal_preds = [PredicateAtom("clear", ["b"])]
planner = LiftedPlanner(domain)

print(f"Testing ORIGINAL version (commit 8b362d9)")
print(f"Goal: {[str(p) for p in goal_preds]}\n")

result = planner.explore_from_goal(goal_preds, max_states=15000)

print(f"\nORIGINAL VERSION RESULT:")
print(f"  States: {len(result['states']):,}")
print(f"  Transitions: {len(result['transitions']):,}")

depth_counts = {}
for state in result['states']:
    d = state.depth
    depth_counts[d] = depth_counts.get(d, 0) + 1

print(f"\nDepth distribution:")
for depth in sorted(depth_counts.keys())[:10]:
    count = depth_counts[depth]
    pct = (count / len(result['states'])) * 100
    print(f"  Depth {depth}: {count:,} ({pct:.1f}%)")
