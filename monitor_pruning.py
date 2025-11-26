#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

print("=" * 80)
print("RUNTIME MONITORING: Count how many states are pruned by mutex")
print("=" * 80)

domain_file = Path(__file__).parent / "src" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))
planner = BackwardSearchPlanner(domain, domain_path=str(domain_file))

# Patch the _check_no_mutex_violations method to count calls
original_check = planner._check_no_mutex_violations
pruned_count = [0]
checked_count = [0]

def monitored_check(predicates):
    checked_count[0] += 1
    result = original_check(predicates)
    if not result:  # Mutex violation detected
        pruned_count[0] += 1
    return result

planner._check_no_mutex_violations = monitored_check

print("\n[1] Running small search with 2 objects...")
goal_predicates = [PredicateAtom("on", ["a", "b"], negated=True)]
objects = ['a', 'b']

import time
start = time.time()
state_graph = planner.search(
    goal_predicates=goal_predicates,
    max_states=5000,  # Small limit for quick test
    max_objects=2,
    objects=objects
)
elapsed = time.time() - start

print(f"\n[2] Results:")
print(f"  Time: {elapsed:.2f}s")
print(f"  States explored: {planner._states_explored:,}")
print(f"  Mutex checks performed: {checked_count[0]:,}")
print(f"  States pruned by mutex: {pruned_count[0]:,}")

if pruned_count[0] > 0:
    prune_rate = 100 * pruned_count[0] / checked_count[0]
    print(f"  Pruning rate: {prune_rate:.1f}%")
    print(f"\n✅ Mutex pruning IS WORKING!")
else:
    print(f"\n❌ NO STATES WERE PRUNED - Mutex checking is not pruning anything!")
    
print("=" * 80)
