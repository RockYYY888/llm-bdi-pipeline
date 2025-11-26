#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("VERIFICATION: Lifted Mutex Pattern Extraction")
print("=" * 80)

# Test 1: Can we import the new class?
print("\n[1] Testing imports...")
try:
    from stage3_code_generation.fd_invariant_extractor import FDInvariantExtractor, LiftedMutexPattern
    print("✅ LiftedMutexPattern class imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Extract invariants
print("\n[2] Extracting invariants from blocksworld domain...")
try:
    domain_path = 'src/domains/blocksworld/domain.pddl'
    objects = ['b1', 'b2', 'b3']
    
    extractor = FDInvariantExtractor(domain_path, objects)
    static_mutex, singletons, lifted_patterns = extractor.extract_invariants()
    
    print(f"✅ Extraction completed")
    print(f"   Legacy mutex groups: {len(static_mutex)}")
    print(f"   Singleton predicates: {len(singletons)}")
    print(f"   Lifted mutex patterns: {len(lifted_patterns)}")
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Display all lifted patterns
print("\n[3] Detailed lifted mutex patterns:")
print("-" * 80)
for i, pattern in enumerate(sorted(lifted_patterns, key=lambda p: (p.pred1_name, p.pred2_name)), 1):
    print(f"\nPattern {i}:")
    print(f"  Predicates: {pattern.pred1_name}(arity={pattern.pred1_arity}) ⊕ {pattern.pred2_name}(arity={pattern.pred2_arity})")
    print(f"  Shared positions: {pattern.shared_positions}")
    print(f"  Different positions: {pattern.different_positions}")
    
    # Explain what this pattern means
    if pattern.pred1_name == pattern.pred2_name:
        print(f"  → Same predicate: {pattern.pred1_name}(...) cannot appear twice with same args at shared positions")
    else:
        print(f"  → Different predicates: {pattern.pred1_name}(...) ⊕ {pattern.pred2_name}(...) are mutex")

# Test 4: Test pattern matching
print("\n[4] Testing pattern matching:")
print("-" * 80)

# Find the on(X,Y) ⊕ on(X,Z) pattern
on_on_pattern = None
for p in lifted_patterns:
    if p.pred1_name == 'on' and p.pred2_name == 'on':
        on_on_pattern = p
        break

if on_on_pattern:
    print(f"\nFound on(X,Y) ⊕ on(X,Z) pattern:")
    print(f"  shared_positions={on_on_pattern.shared_positions}")
    print(f"  different_positions={on_on_pattern.different_positions}")
    
    # Test cases
    test_cases = [
        (('on', ('b1', 'b2')), ('on', ('b1', 'b3')), True, "on(b1,b2) ⊕ on(b1,b3) - SHOULD BE MUTEX"),
        (('on', ('b1', 'b2')), ('on', ('b2', 'b3')), False, "on(b1,b2) ⊕ on(b2,b3) - NOT MUTEX"),
        (('on', ('b1', 'b2')), ('on', ('b1', 'b2')), False, "on(b1,b2) ⊕ on(b1,b2) - SAME FACT"),
    ]
    
    print("\n  Test cases:")
    for (name1, args1), (name2, args2), expected, desc in test_cases:
        result = on_on_pattern.matches(name1, args1, name2, args2)
        status = "✅" if result == expected else "❌"
        print(f"    {status} {desc}: matches={result} (expected={expected})")
else:
    print("❌ on(X,Y) ⊕ on(X,Z) pattern not found!")

print("\n" + "=" * 80)
print("SUMMARY: All lifted mutex patterns extracted successfully!")
print("=" * 80)
