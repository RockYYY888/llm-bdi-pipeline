#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser
from collections import defaultdict

print("=" * 80)
print("DETAILED PRUNING STATISTICS")
print("=" * 80)

domain_file = Path(__file__).parent / "src" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))
planner = BackwardSearchPlanner(domain, domain_path=str(domain_file))

# Track which patterns are triggering
pattern_triggers = defaultdict(int)
total_checks = [0]
total_pruned = [0]

original_check = planner._check_no_mutex_violations
def monitored_check(predicates):
    total_checks[0] += 1
    
    # Inline the lifted pattern logic with tracking
    positive_preds = [p for p in predicates if not p.negated]
    
    for i, pred1 in enumerate(positive_preds):
        for pred2 in positive_preds[i+1:]:
            if pred1 == pred2:
                continue
            
            for pattern in planner.lifted_mutex_patterns:
                if pattern.matches(pred1.name, pred1.args, pred2.name, pred2.args):
                    # Record which pattern triggered
                    key = f"{pattern.pred1_name}/{pattern.pred1_arity} ⊕ {pattern.pred2_name}/{pattern.pred2_arity}"
                    pattern_triggers[key] += 1
                    total_pruned[0] += 1
                    return False
    
    return True

planner._check_no_mutex_violations = monitored_check

print(f"\n[1] Available patterns:")
for i, p in enumerate(planner.lifted_mutex_patterns, 1):
    print(f"  {i}. {p.pred1_name}/{p.pred1_arity} ⊕ {p.pred2_name}/{p.pred2_arity}")
    print(f"     shared={p.shared_positions}, diff={p.different_positions}")

print(f"\n[2] Running search (max 1000 states)...")
goal_predicates = [PredicateAtom("on", ["a", "b"], negated=True)]
objects = ['a', 'b']

state_graph = planner.search(
    goal_predicates=goal_predicates,
    max_states=1000,
    max_objects=2,
    objects=objects
)

print(f"\n[3] Pruning statistics:")
print(f"  Total mutex checks: {total_checks[0]:,}")
print(f"  Total states pruned: {total_pruned[0]:,}")
if total_checks[0] > 0:
    print(f"  Pruning rate: {100*total_pruned[0]/total_checks[0]:.1f}%")

print(f"\n[4] Pattern trigger frequency:")
if pattern_triggers:
    sorted_patterns = sorted(pattern_triggers.items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns:
        pct = 100 * count / total_pruned[0] if total_pruned[0] > 0 else 0
        print(f"  {pattern}: {count:,} ({pct:.1f}%)")
else:
    print(f"  ❌ NO PATTERNS TRIGGERED!")

print("=" * 80)
