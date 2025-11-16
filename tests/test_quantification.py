"""
Test Quantification功能
"""
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent / "src")
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from stage3_code_generation.state_space import PredicateAtom
from stage3_code_generation.quantified_predicate import detect_quantifiable_pattern, Quantifier
from stage3_code_generation.abstract_state import AbstractState, ConstraintSet

def test_quantification_detection():
    """Test that quantification detection works"""
    print("="*80)
    print("Test: Quantification Detection")
    print("="*80)

    # Create predicates with pattern: on(?X, b) repeated
    predicates = {
        PredicateAtom("on", ("?V1", "b")),
        PredicateAtom("on", ("?V2", "b")),
        PredicateAtom("on", ("?V3", "b")),
        PredicateAtom("clear", ("b",)),
        PredicateAtom("handempty", ())
    }

    print(f"\nInput predicates ({len(predicates)}): ")
    for p in sorted(predicates, key=str):
        print(f"  {p}")

    # Detect quantifiable patterns
    quantified = detect_quantifiable_pattern(predicates, min_instances=2)

    print(f"\nDetected {len(quantified)} quantified patterns:")
    for qp in quantified:
        print(f"  {qp}")

    # Check results
    assert len(quantified) > 0, "Should detect at least one quantified pattern"

    # First quantified predicate should be for on(??, b)
    qp = quantified[0]
    assert qp.quantifier == Quantifier.EXISTS
    assert qp.formula.name == "on"
    print(f"\n✓ Quantification detection works!")
    print(f"  Formula: {qp.formula}")
    print(f"  Quantified vars: {qp.variables}")
    print(f"  Count bound: {qp.count_bound}")

def test_quantified_state_creation():
    """Test creating states with quantified predicates"""
    print("\n" + "="*80)
    print("Test: Quantified State Creation and Deduplication")
    print("="*80)

    # Create state without quantification
    concrete_preds = {
        PredicateAtom("on", ("?V1", "b")),
        PredicateAtom("on", ("?V2", "b")),
        PredicateAtom("on", ("?V3", "b")),
        PredicateAtom("clear", ("b",)),
        PredicateAtom("handempty", ())
    }

    state_concrete = AbstractState(concrete_preds, ConstraintSet())

    print(f"\nConcrete state ({len(state_concrete.predicates)} predicates):")
    print(f"  {state_concrete}")

    # Now test with quantified predicates
    from stage3_code_generation.lifted_planner import LiftedPlanner
    from utils.pddl_parser import PDDLDomain, PDDLAction

    # Create minimal domain for testing
    domain = PDDLDomain("blocksworld", [], [], [])

    planner = LiftedPlanner(domain)

    # Apply quantification
    state_quantified = planner._detect_and_quantify_state(state_concrete, min_instances=2)

    print(f"\nQuantified state:")
    print(f"  Concrete predicates: {len(state_quantified.predicates)}")
    for p in sorted(state_quantified.predicates, key=str):
        print(f"    {p}")
    print(f"  Quantified predicates: {len(state_quantified.quantified_predicates or [])}")
    for qp in sorted(state_quantified.quantified_predicates or [], key=str):
        print(f"    {qp}")

    # Should have fewer concrete predicates after quantification
    assert len(state_quantified.predicates) < len(state_concrete.predicates), \
        f"Expected fewer predicates after quantification: {len(state_quantified.predicates)} vs {len(state_concrete.predicates)}"

    print(f"\n✓ Quantification reduces predicates: {len(state_concrete.predicates)} → {len(state_quantified.predicates)}")

    # Check state keys are different
    key_concrete = planner._state_key(state_concrete)
    key_quantified = planner._state_key(state_quantified)

    print(f"\n  State key (concrete): {len(str(key_concrete))} chars")
    print(f"  State key (quantified): {len(str(key_quantified))} chars")

    assert key_concrete != key_quantified, "State keys should be different"
    print(f"✓ State keys are distinct")

if __name__ == "__main__":
    test_quantification_detection()
    test_quantified_state_creation()
    print("\n" + "="*80)
    print("All quantification tests passed!")
    print("="*80)
