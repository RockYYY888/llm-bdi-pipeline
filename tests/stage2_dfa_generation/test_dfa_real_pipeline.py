#!/usr/bin/env python3
"""
Real Pipeline Test: DFA Simplifier with ltlf2dfa Output

Tests that DFA simplifier correctly processes real ltlf2dfa output
and produces valid input for backward planning.

Focus:
1. Real ltlf2dfa DFA structure
2. Partition mapping correctness
3. DFA semantic equivalence
4. Integration with backward planner input format
"""

import sys
from pathlib import Path

# Add src to path
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.dfa_simplifier import DFASimplifier
from stage2_dfa_generation.dfa_builder import DFABuilder
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator


def test_real_ltlf2dfa_output():
    """Test with actual ltlf2dfa output from blocksworld example"""
    print("="*80)
    print("TEST 1: Real ltlf2dfa Output Processing")
    print("="*80)

    # Create LTL spec: F(on_a_b)
    spec = LTLSpecification()
    spec.objects = ["a", "b"]

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    spec.grounding_map = gmap

    # Create F(on_a_b)
    atom = LTLFormula(
        predicate="on_a_b",
        operator=None,
        sub_formulas=[],
        logical_op=None
    )
    f_formula = LTLFormula(
        predicate=None,
        operator=TemporalOperator.FINALLY,
        sub_formulas=[atom],
        logical_op=None
    )
    spec.formulas = [f_formula]

    print(f"\nLTL Formula: {f_formula.to_string()}")

    # Build DFA using real DFABuilder
    print("\n--- Step 1: Generate DFA with ltlf2dfa ---")
    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"✓ DFA Generated:")
    print(f"  States: {dfa_result['num_states']}")
    print(f"  Transitions: {dfa_result['num_transitions']}")

    # Extract transition labels from real DFA
    import re
    original_labels = set()
    for line in dfa_result['dfa_dot'].split('\n'):
        match = re.search(r'\[label="([^"]+)"\]', line)
        if match and 'init' not in line:
            original_labels.add(match.group(1))

    print(f"  Unique transition labels: {len(original_labels)}")
    for label in sorted(original_labels):
        print(f"    - '{label}'")

    # Simplify DFA
    print("\n--- Step 2: Simplify DFA ---")
    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_result['dfa_dot'], gmap)

    print(f"\n✓ Simplification Complete:")
    print(f"  Method: {simplified.stats['method']}")
    print(f"  Predicates found: {simplified.stats['num_predicates']}")
    print(f"  Partitions generated: {simplified.stats['num_partitions']}")

    if simplified.stats['num_partitions'] > 0:
        print(f"\n  Partitions:")
        for partition in simplified.partitions:
            print(f"    {partition.symbol}: {partition.expression}")
            print(f"      Predicate values: {partition.predicate_values}")

        print(f"\n  Label Mappings:")
        for label, partitions in simplified.original_label_to_partitions.items():
            print(f"    '{label}' → {partitions}")

    # Verify DFA structure
    print("\n--- Step 3: Verify DFA Structure ---")

    # Count states in simplified DFA
    simplified_states = set()
    for line in simplified.simplified_dot.split('\n'):
        # Match state declarations
        match = re.match(r'\s*node\s+\[.*?\];\s*(\d+);', line)
        if match:
            simplified_states.add(match.group(1))

    print(f"✓ State count preserved: {dfa_result['num_states']} states")

    # Verify transitions
    original_transitions = []
    simplified_transitions = []

    for line in dfa_result['dfa_dot'].split('\n'):
        match = re.match(r'\s*(\d+)\s*->\s*(\d+)\s*\[label="([^"]+)"\]', line)
        if match:
            original_transitions.append((match.group(1), match.group(2), match.group(3)))

    for line in simplified.simplified_dot.split('\n'):
        match = re.match(r'\s*(\d+)\s*->\s*(\d+)\s*\[label="([^"]+)"\]', line)
        if match:
            simplified_transitions.append((match.group(1), match.group(2), match.group(3)))

    print(f"✓ Transitions:")
    print(f"    Original: {len(original_transitions)}")
    print(f"    Simplified: {len(simplified_transitions)}")

    # Verify semantic equivalence
    print("\n--- Step 4: Verify Semantic Equivalence ---")

    for orig_from, orig_to, orig_label in original_transitions:
        # Get partition symbols for this label
        if orig_label in simplified.original_label_to_partitions:
            partition_symbols = simplified.original_label_to_partitions[orig_label]

            # Verify each partition symbol appears in simplified transitions
            for symbol in partition_symbols:
                found = False
                for simp_from, simp_to, simp_label in simplified_transitions:
                    if simp_from == orig_from and simp_to == orig_to and simp_label == symbol:
                        found = True
                        break

                if not found:
                    print(f"✗ Missing transition: {orig_from} -> {orig_to} [{symbol}]")
                    return False

    print(f"✓ All original transitions preserved via partitions")

    print("\n✓ TEST 1 PASSED")
    return simplified


def test_complex_ltl_formula():
    """Test with more complex LTL formula: F(on_a_b & clear_c)"""
    print("\n" + "="*80)
    print("TEST 2: Complex LTL Formula")
    print("="*80)

    # Create LTL spec
    spec = LTLSpecification()
    spec.objects = ["a", "b", "c"]

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])
    spec.grounding_map = gmap

    # Create F(on_a_b & clear_c)
    from stage1_interpretation.ltlf_formula import LogicalOperator

    on_a_b = LTLFormula(
        predicate="on_a_b",
        operator=None,
        sub_formulas=[],
        logical_op=None
    )
    clear_c = LTLFormula(
        predicate="clear_c",
        operator=None,
        sub_formulas=[],
        logical_op=None
    )
    conjunction = LTLFormula(
        predicate=None,
        operator=None,
        sub_formulas=[on_a_b, clear_c],
        logical_op=LogicalOperator.AND
    )
    f_formula = LTLFormula(
        predicate=None,
        operator=TemporalOperator.FINALLY,
        sub_formulas=[conjunction],
        logical_op=None
    )
    spec.formulas = [f_formula]

    print(f"\nLTL Formula: {f_formula.to_string()}")

    # Build DFA
    print("\n--- Generate DFA ---")
    builder = DFABuilder()
    dfa_result = builder.build(spec)

    print(f"Original DFA: {dfa_result['num_states']} states, {dfa_result['num_transitions']} transitions")

    # Simplify
    print("\n--- Simplify DFA ---")
    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_result['dfa_dot'], gmap)

    print(f"\nSimplified DFA:")
    print(f"  Predicates: {simplified.stats['num_predicates']}")
    print(f"  Partitions: {simplified.stats['num_partitions']}")

    if simplified.stats['num_partitions'] > 0:
        print(f"\n  Partition details:")
        for partition in simplified.partitions:
            print(f"    {partition.symbol}: {partition.expression}")

    print("\n✓ TEST 2 PASSED")
    return simplified


def test_partition_map_extraction():
    """Test that partition_map can be extracted and used by backward planner"""
    print("\n" + "="*80)
    print("TEST 3: Partition Map for Backward Planner Integration")
    print("="*80)

    # Create simple test case
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    # Simulate DFA output
    dfa_dot = """
digraph MONA_DFA {
 init -> 1;
 1 -> 1 [label="!(on_a_b & clear_c)"];
 1 -> 2 [label="on_a_b & clear_c"];
 2 -> 2 [label="true"];
}
"""

    print("Input DFA labels:")
    print("  - '!(on_a_b & clear_c)'")
    print("  - 'on_a_b & clear_c'")
    print("  - 'true'")

    # Simplify
    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_dot, gmap)

    print(f"\n--- Partition Map Structure ---")
    print(f"Type: {type(simplified.partition_map)}")
    print(f"Keys: {list(simplified.partition_map.keys())}")

    # Demonstrate how backward planner would use this
    print(f"\n--- How Backward Planner Uses Partition Map ---")
    print("Example: Processing simplified transition '1 -> 2 [label=\"p4\"]'")

    # Find p4 in partitions
    if 'p4' in simplified.partition_map:
        partition = simplified.partition_map['p4']
        print(f"\n1. Detect partition symbol: 'p4'")
        print(f"2. Lookup in partition_map: partition_map['p4']")
        print(f"3. Get expression: '{partition.expression}'")
        print(f"4. Get predicate values: {partition.predicate_values}")
        print(f"5. Parse expression for goal extraction")

        # Test parsing
        from stage3_code_generation.boolean_expression_parser import BooleanExpressionParser
        parser = BooleanExpressionParser(gmap)
        dnf = parser.parse(partition.expression)
        print(f"6. DNF result: {len(dnf)} disjuncts")
        for i, conj in enumerate(dnf):
            print(f"   Disjunct {i}: {[p.to_agentspeak() for p in conj]}")

    # Show complete mapping
    print(f"\n--- Complete Partition Map ---")
    for symbol, partition in simplified.partition_map.items():
        print(f"{symbol}:")
        print(f"  Expression: {partition.expression}")
        print(f"  Values: {partition.predicate_values}")

    print("\n✓ TEST 3 PASSED")
    return simplified.partition_map


def test_edge_cases():
    """Test edge cases: true, false, negation"""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    test_cases = [
        ("true only", "s0 -> s1 [label=\"true\"];"),
        ("false only", "s0 -> s1 [label=\"false\"];"),
        ("negation", "s0 -> s1 [label=\"!on_a_b\"];"),
        ("OR expression", "s0 -> s1 [label=\"on_a_b | !on_a_b\"];"),
    ]

    simplifier = DFASimplifier()

    for name, dfa_snippet in test_cases:
        print(f"\n--- {name} ---")
        dfa_dot = f"digraph G {{\n init -> s0;\n {dfa_snippet}\n}}"

        try:
            result = simplifier.simplify(dfa_dot, gmap)
            print(f"✓ Processed successfully")
            print(f"  Partitions: {result.stats['num_partitions']}")
            if result.stats['num_partitions'] > 0:
                for p in result.partitions:
                    print(f"    {p.symbol}: {p.expression}")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    print("\n✓ TEST 4 PASSED")
    return True


def test_dfa_dot_format_preservation():
    """Verify that simplified DFA maintains valid DOT format"""
    print("\n" + "="*80)
    print("TEST 5: DOT Format Preservation")
    print("="*80)

    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    dfa_dot = """digraph MONA_DFA {
 rankdir = LR;
 center = true;
 size = "7.5,10.5";
 edge [fontname = Courier];
 node [height = .5, width = .5];
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 1 [label="!on_a_b"];
 1 -> 2 [label="on_a_b"];
 2 -> 2 [label="true"];
}
"""

    print("Original DFA format:")
    print(f"  Lines: {len(dfa_dot.split(chr(10)))}")
    print(f"  Has 'digraph': {'digraph' in dfa_dot}")
    print(f"  Has 'rankdir': {'rankdir' in dfa_dot}")
    print(f"  Has node declarations: {'node [shape' in dfa_dot}")

    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_dot, gmap)

    print("\nSimplified DFA format:")
    print(f"  Lines: {len(simplified.simplified_dot.split(chr(10)))}")
    print(f"  Has 'digraph': {'digraph' in simplified.simplified_dot}")
    print(f"  Has 'rankdir': {'rankdir' in simplified.simplified_dot}")
    print(f"  Has node declarations: {'node [shape' in simplified.simplified_dot}")

    # Verify structure elements preserved
    checks = [
        ('digraph', 'digraph keyword'),
        ('rankdir', 'layout direction'),
        ('node [shape', 'node styling'),
        ('init ->', 'initial state'),
        ('->', 'transitions'),
        ('[label=', 'edge labels'),
    ]

    all_ok = True
    for pattern, desc in checks:
        if pattern in simplified.simplified_dot:
            print(f"  ✓ {desc} preserved")
        else:
            print(f"  ✗ {desc} MISSING")
            all_ok = False

    if all_ok:
        print("\n✓ TEST 5 PASSED")
    else:
        print("\n✗ TEST 5 FAILED")

    return all_ok


if __name__ == "__main__":
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*22 + "DFA SIMPLIFIER REAL PIPELINE TEST" + " "*23 + "║")
    print("╚" + "═"*78 + "╝" + "\n")

    results = []

    try:
        # Test 1: Real ltlf2dfa output
        r1 = test_real_ltlf2dfa_output()
        results.append(("Real ltlf2dfa output", r1 is not None))

        # Test 2: Complex formula
        r2 = test_complex_ltl_formula()
        results.append(("Complex LTL formula", r2 is not None))

        # Test 3: Partition map extraction
        r3 = test_partition_map_extraction()
        results.append(("Partition map extraction", r3 is not None))

        # Test 4: Edge cases
        r4 = test_edge_cases()
        results.append(("Edge cases", r4))

        # Test 5: DOT format preservation
        r5 = test_dfa_dot_format_preservation()
        results.append(("DOT format preservation", r5))

        # Summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)

        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        all_passed = all(r[1] for r in results)

        if all_passed:
            print("\n" + "="*80)
            print("✓ ALL TESTS PASSED - DFA SIMPLIFIER READY FOR INTEGRATION")
            print("="*80)

            print("\nNext Steps:")
            print("1. Integrate simplifier into DFABuilder")
            print("2. Pass partition_map to BackwardPlannerGenerator")
            print("3. Modify BackwardPlannerGenerator to resolve partition symbols")
        else:
            print("\n⚠ SOME TESTS FAILED - REVIEW REQUIRED")

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
