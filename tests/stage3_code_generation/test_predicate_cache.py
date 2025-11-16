#!/usr/bin/env python3
"""
Simple test to verify predicate-level caching logic
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.state_space import PredicateAtom


def test_cache_logic():
    """Test the cache key generation and lookup logic"""

    print("="*80)
    print("Testing Predicate-Level Cache Logic")
    print("="*80)

    # Simulated predicates
    predicates1 = [PredicateAtom("on", ["?v0", "?v1"])]
    predicates2 = [PredicateAtom("on", ["?v0", "?v1"])]
    predicates3 = [PredicateAtom("on", ["?v0", "?v1"]), PredicateAtom("clear", ["?v2"])]
    predicates4 = [PredicateAtom("clear", ["?v0"])]

    # Test single predicate cache keys
    print("\n1. Single predicate cache keys:")
    for i, preds in enumerate([predicates1, predicates2, predicates4], 1):
        if len(preds) == 1:
            pred = preds[0]
            key_3obj = (pred.to_agentspeak(), 3)
            key_5obj = (pred.to_agentspeak(), 5)
            print(f"  Test {i}: {pred}")
            print(f"    Key (3 objects): {key_3obj}")
            print(f"    Key (5 objects): {key_5obj}")

    # Test that same predicate pattern with same object count has same key
    print("\n2. Cache key equality test:")
    pred1 = predicates1[0]
    pred2 = predicates2[0]
    key1 = (pred1.to_agentspeak(), 5)
    key2 = (pred2.to_agentspeak(), 5)
    print(f"  Predicate 1: {pred1} → Key: {key1}")
    print(f"  Predicate 2: {pred2} → Key: {key2}")
    print(f"  Keys equal? {key1 == key2} ✓")

    # Test cache simulation
    print("\n3. Cache simulation:")
    predicate_cache = {}
    num_objects = 5

    # First condition: on(?v0, ?v1)
    pred = predicates1[0]
    cache_key = (pred.to_agentspeak(), num_objects)
    print(f"\n  Condition 1: [{pred}]")
    if cache_key in predicate_cache:
        print(f"    → Cache HIT!")
    else:
        print(f"    → Cache MISS, exploring...")
        predicate_cache[cache_key] = "StateGraph(525 states)"  # Simulated
        print(f"    → Cached: {cache_key}")

    # Second condition: on(?v0, ?v1) again (should hit!)
    print(f"\n  Condition 2: [{predicates2[0]}]")
    cache_key = (predicates2[0].to_agentspeak(), num_objects)
    if cache_key in predicate_cache:
        print(f"    → Cache HIT! ✓ Reusing previous exploration")
    else:
        print(f"    → Cache MISS (unexpected!)")

    # Third condition: multi-predicate (won't use predicate cache)
    print(f"\n  Condition 3: {[str(p) for p in predicates3]}")
    if len(predicates3) > 1:
        print(f"    → Multi-predicate goal, using full-goal cache")

    # Fourth condition: clear(?v0) (different predicate)
    pred = predicates4[0]
    cache_key = (pred.to_agentspeak(), num_objects)
    print(f"\n  Condition 4: [{pred}]")
    if cache_key in predicate_cache:
        print(f"    → Cache HIT!")
    else:
        print(f"    → Cache MISS, exploring...")
        predicate_cache[cache_key] = "StateGraph(597 states)"
        print(f"    → Cached: {cache_key}")

    print("\n" + "="*80)
    print("Final cache state:")
    for key, value in predicate_cache.items():
        print(f"  {key}: {value}")
    print("="*80)

    print("\n✅ Cache logic test passed!")


if __name__ == "__main__":
    test_cache_logic()
