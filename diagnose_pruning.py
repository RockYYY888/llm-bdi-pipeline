#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage3_code_generation.backward_search_refactored import BackwardSearchPlanner
from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLParser

print("=" * 80)
print("DIAGNOSIS: Are lifted mutex patterns actually pruning states?")
print("=" * 80)

domain_file = Path(__file__).parent / "src" / "domains" / "blocksworld" / "domain.pddl"
domain = PDDLParser.parse_domain(str(domain_file))

print("\n[1] Creating planner...")
planner = BackwardSearchPlanner(domain, domain_path=str(domain_file))

print(f"\n[2] Checking if lifted patterns were loaded:")
if hasattr(planner, 'lifted_mutex_patterns'):
    print(f"  ✅ lifted_mutex_patterns exists: {len(planner.lifted_mutex_patterns)} patterns")
    for p in planner.lifted_mutex_patterns:
        print(f"    - {p.pred1_name}/{p.pred1_arity} ⊕ {p.pred2_name}/{p.pred2_arity}")
else:
    print(f"  ❌ lifted_mutex_patterns NOT FOUND!")
    
print(f"\n[3] Testing mutex checking directly:")
# Test case: should holding(b1) and holding(b2) be mutex?
from stage3_code_generation.state_space import PredicateAtom

test_preds = {
    PredicateAtom("holding", ["b1"], negated=False),
    PredicateAtom("holding", ["b2"], negated=False)
}

result = planner._check_no_mutex_violations(test_preds)
print(f"  Test: holding(b1) + holding(b2)")
print(f"  Result: {'✅ NOT MUTEX (WRONG!)' if result else '❌ MUTEX (CORRECT!)'}")

# Test case 2: should on(b1,b2) and on(b1,b3) be mutex?
test_preds2 = {
    PredicateAtom("on", ["b1", "b2"], negated=False),
    PredicateAtom("on", ["b1", "b3"], negated=False)
}

result2 = planner._check_no_mutex_violations(test_preds2)
print(f"\n  Test: on(b1,b2) + on(b1,b3)")
print(f"  Result: {'✅ NOT MUTEX (WRONG!)' if result2 else '❌ MUTEX (CORRECT!)'}")

# Test case 3: should on(b1,b2) and on(b2,b3) NOT be mutex?
test_preds3 = {
    PredicateAtom("on", ["b1", "b2"], negated=False),
    PredicateAtom("on", ["b2", "b3"], negated=False)
}

result3 = planner._check_no_mutex_violations(test_preds3)
print(f"\n  Test: on(b1,b2) + on(b2,b3)")
print(f"  Result: {'✅ NOT MUTEX (CORRECT!)' if result3 else '❌ MUTEX (WRONG!)'}")

print("\n" + "=" * 80)
if not result and not result2 and result3:
    print("✅ Mutex checking is working correctly!")
else:
    print("❌ Mutex checking is NOT working!")
    print("\nDEBUG: Checking _check_no_mutex_violations implementation...")
    
    # Check which method is being called
    import inspect
    source = inspect.getsource(planner._check_no_mutex_violations)
    if 'lifted_mutex_patterns' in source:
        print("  ✓ Method uses lifted_mutex_patterns")
    else:
        print("  ✗ Method does NOT use lifted_mutex_patterns")
        
    if 'legacy' in source.lower():
        print("  ⚠ Method has legacy fallback")
        
print("=" * 80)
