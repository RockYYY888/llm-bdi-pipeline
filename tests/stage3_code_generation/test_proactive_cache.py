#!/usr/bin/env python3
"""
Test proactive predicate caching from multi-predicate goals
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.variable_normalizer import VariableNormalizer, VariableMapping
from stage3_code_generation.forward_planner import ForwardStatePlanner
from utils.pddl_parser import PDDLParser


def test_proactive_caching_logic():
    """Test that proactive caching logic works correctly"""
    print("=" * 80)
    print("Test: Proactive Predicate Caching from Multi-Predicate Goals")
    print("=" * 80)

    # Load domain
    domain_file = Path(__file__).parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return False

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"✓ Loaded domain: {domain.name}\n")

    # Setup
    objects = ["a", "b", "c"]
    normalizer = VariableNormalizer(domain, objects)
    variables = normalizer.get_variable_list(len(objects))

    # Simulate two-tier cache
    predicate_cache = {}
    full_goal_cache = {}

    print("Scenario: Simulate proactive caching for multi-predicate goal")
    print("-" * 80)

    # Step 1: Multi-predicate goal (simulating Condition 1)
    print("\n[Step 1] Explore multi-predicate goal: [on(a,b), clear(c)]")

    goal1 = [PredicateAtom("on", ["a", "b"]), PredicateAtom("clear", ["c"])]
    normalized1, mapping1 = normalizer.normalize_predicates(goal1)
    print(f"  Normalized: {[str(p) for p in normalized1]}")
    print(f"  Mapping: {mapping1.obj_to_var}")

    # Explore multi-predicate goal
    planner1 = ForwardStatePlanner(domain, variables, use_variables=True)
    state_graph1 = planner1.explore_from_goal(normalized1)
    print(f"  Explored: {state_graph1}")

    # Cache full goal
    goal_key1 = normalizer.serialize_goal(normalized1)
    full_goal_cache[goal_key1] = (state_graph1, mapping1)
    print(f"  ✓ Cached in full-goal cache: {goal_key1}")

    # PROACTIVE CACHING: Also explore and cache individual predicates
    print(f"\n  [Proactive Caching] Exploring individual predicates...")
    for pred in normalized1:
        pred_key = (pred.to_agentspeak(), len(objects))
        if pred_key not in predicate_cache:
            print(f"    → Exploring single predicate: {pred}")
            single_planner = ForwardStatePlanner(domain, variables, use_variables=True)
            single_graph = single_planner.explore_from_goal([pred])

            # Create minimal mapping
            single_mapping_obj_to_var = {obj: var for obj, var in mapping1.obj_to_var.items()
                                        if var in pred.args}
            single_mapping_var_to_obj = {var: obj for var, obj in mapping1.var_to_obj.items()
                                        if var in pred.args}
            single_mapping = VariableMapping(single_mapping_obj_to_var, single_mapping_var_to_obj)

            predicate_cache[pred_key] = (single_graph, single_mapping)
            print(f"      ✓ Cached: {pred_key[0]} → {single_graph}")

    print(f"\n  Cache state after Step 1:")
    print(f"    Predicate cache: {len(predicate_cache)} entries")
    for key, (graph, _) in predicate_cache.items():
        print(f"      - {key[0]}: {len(graph.states)} states")
    print(f"    Full-goal cache: {len(full_goal_cache)} entries")

    # Step 2: Encounter a single predicate that was proactively cached
    print("\n" + "=" * 80)
    print("[Step 2] Encounter single predicate: on(c,a) (simulating Condition 2)")
    print("(Using different objects from step 1, but same predicate pattern)")

    goal2 = [PredicateAtom("on", ["c", "a"])]  # Different objects, but should normalize to on(?v0,?v1)
    normalized2, mapping2 = normalizer.normalize_predicates(goal2)
    print(f"  Normalized: {[str(p) for p in normalized2]}")
    print(f"  Mapping: {mapping2.obj_to_var}")

    # Check predicate cache
    pred_key2 = (normalized2[0].to_agentspeak(), len(objects))
    print(f"  Cache key: {pred_key2}")

    if pred_key2 in predicate_cache:
        cached_graph, cached_mapping = predicate_cache[pred_key2]
        print(f"  ✓✓✓ CACHE HIT! ✓✓✓")
        print(f"  Reused state graph: {cached_graph}")
        print(f"  Saved exploration of {len(cached_graph.states)} states!")
        return True
    else:
        print(f"  ✗ CACHE MISS (unexpected)")
        return False


if __name__ == "__main__":
    success = test_proactive_caching_logic()
    if success:
        print("\n" + "=" * 80)
        print("✅ TEST PASSED: Proactive caching enables cache hits!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        sys.exit(1)
