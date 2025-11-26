#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

print("Testing LIFTED MUTEX PATTERNS Implementation")
print("=" * 80)

domain_file = Path(__file__).parent / "src" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))

print("\n[1] Creating planner with lifted mutex patterns...")
planner = BackwardSearchPlanner(domain, domain_path=str(domain_file))

print(f"\n[2] Lifted patterns extracted: {len(planner.lifted_mutex_patterns)}")
for pattern in sorted(planner.lifted_mutex_patterns, key=lambda p: (p.pred1_name, p.pred2_name)):
    print(f"  • {pattern.pred1_name}/{pattern.pred1_arity} ⊕ {pattern.pred2_name}/{pattern.pred2_arity}")
    print(f"    shared={pattern.shared_positions}, diff={pattern.different_positions}")

print("\n[3] Testing ~on(a, b) with 2 objects - should prune effectively now")
goal_predicates = [PredicateAtom("on", ["a", "b"], negated=True)]
objects = ['a', 'b']

import time
start = time.time()
state_graph = planner.search(
    goal_predicates=goal_predicates,
    max_states=200000,
    max_objects=2,
    objects=objects
)
elapsed = time.time() - start

print(f"\n[4] Results:")
print(f"  Time: {elapsed:.2f}s")
print(f"  States explored: {planner._states_explored:,}")
print(f"  Total states in graph: {len(state_graph.states):,}")
print(f"  Transitions: {len(state_graph.transitions):,}")

if planner._states_explored < 200000:
    print("\n✅ SUCCESS: Search completed before hitting limit!")
    print(f"✅ Pruning reduced exploration by {100*(1 - planner._states_explored/200000):.1f}%")
else:
    print("\n❌ PROBLEM: Still hit the 200k state limit")
