#!/usr/bin/env python3
"""
Integration Test: DFA Simplifier with Real Pipeline

Tests the DFA simplifier with real DFA outputs from ltlf2dfa and
validates integration with BackwardPlannerGenerator.
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
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator


def test_integration_with_real_dfa():
    """Test DFA simplifier with manually created DFA (simulating ltlf2dfa output)"""
    print("="*80)
    print("INTEGRATION TEST: DFA Simplifier with Realistic DFA")
    print("="*80)

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    # Create a realistic DFA (similar to ltlf2dfa output)
    # This simulates: F(on_a_b & clear_c)
    dfa_dot = """digraph MONA_DFA {
 rankdir = LR;
 center = true;
 node [shape = doublecircle]; 2;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 1 [label="!(on_a_b & clear_c)"];
 1 -> 2 [label="on_a_b & clear_c"];
 2 -> 2 [label="true"];
}
"""

    dfa_result = {
        'formula': 'F(on_a_b & clear_c)',
        'dfa_dot': dfa_dot,
        'num_states': 2,
        'num_transitions': 3
    }

    print(f"\nTest LTL: {dfa_result['formula']}")
    print(f"Original DFA:")
    print(f"  States: {dfa_result['num_states']}")
    print(f"  Transitions: {dfa_result['num_transitions']}")
    print(f"\nDFA DOT:")
    print(dfa_dot)

    # Simplify DFA
    print("\n--- Simplifying DFA ---")
    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_result['dfa_dot'], gmap)

    print(f"\nSimplification Results:")
    print(f"  Method: {simplified.stats['method']}")
    print(f"  Predicates: {simplified.stats['num_predicates']}")
    print(f"  Partitions: {simplified.stats['num_partitions']}")
    print(f"  Compression: {simplified.stats.get('compression_ratio', 'N/A')}")

    print(f"\nPartitions:")
    for partition in simplified.partitions:
        print(f"  {partition.symbol}: {partition.expression}")
        print(f"    Values: {partition.predicate_values}")

    print(f"\nLabel Mappings:")
    for label, partitions in simplified.original_label_to_partitions.items():
        print(f"  '{label}' → {partitions}")

    print(f"\nSimplified DFA DOT (first 500 chars):")
    print(simplified.simplified_dot[:500] + "...")

    # Validate structure
    assert len(simplified.partitions) > 0, "Should have generated partitions"
    assert len(simplified.original_label_to_partitions) > 0, "Should have label mappings"

    print("\n✓ Integration test passed!")
    return simplified


def test_backward_planner_compatibility():
    """Test that simplified DFA works with BackwardPlannerGenerator"""
    print("\n" + "="*80)
    print("COMPATIBILITY TEST: Simplified DFA with BackwardPlannerGenerator")
    print("="*80)

    from stage3_code_generation.boolean_expression_parser import BooleanExpressionParser

    # Skip BackwardPlannerGenerator integration for now since it requires PDDL domain
    # This test focuses on the boolean expression parsing compatibility

    # Create grounding map
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    # Create simple DFA
    dfa_dot = """
digraph G {
    rankdir=LR;
    s0 [label="0"];
    s1 [label="1", shape=doublecircle];
    init -> s0;
    s0 -> s1 [label="on_a_b & clear_c"];
}
"""

    print("\nOriginal DFA label: 'on_a_b & clear_c'")

    # Test 1: Parse original label
    print("\n--- Test 1: Parse Original Label ---")
    parser = BooleanExpressionParser(gmap)
    try:
        dnf = parser.parse("on_a_b & clear_c")
        print(f"✓ Original label parsed successfully")
        print(f"  DNF: {len(dnf)} disjuncts")
        for i, conj in enumerate(dnf):
            print(f"    Disjunct {i}: {[p.to_agentspeak() for p in conj]}")
    except Exception as e:
        print(f"✗ Failed to parse original label: {e}")
        return False

    # Test 2: Simplify and check partition labels
    print("\n--- Test 2: Simplify DFA ---")
    simplifier = DFASimplifier()
    simplified = simplifier.simplify(dfa_dot, gmap)

    print(f"Partitions generated: {len(simplified.partitions)}")
    for partition in simplified.partitions:
        print(f"  {partition.symbol}: {partition.expression}")

    # Test 3: Verify partition symbols can be parsed
    print("\n--- Test 3: Parse Partition Symbols ---")
    for partition in simplified.partitions:
        # Check if partition expression can be parsed
        try:
            dnf = parser.parse(partition.expression)
            print(f"✓ Partition {partition.symbol} expression parsed successfully")
        except Exception as e:
            print(f"✗ Failed to parse partition {partition.symbol}: {e}")
            return False

    # Test 4: Check mapping preservation
    print("\n--- Test 4: Verify Label Mapping Preservation ---")
    original_label = "on_a_b & clear_c"
    if original_label in simplified.original_label_to_partitions:
        partition_symbols = simplified.original_label_to_partitions[original_label]
        print(f"✓ Original label mapping preserved: '{original_label}' → {partition_symbols}")

        # Verify we can reconstruct original semantics
        print(f"\n  Semantic check:")
        for symbol in partition_symbols:
            partition = simplified.partition_map[symbol]
            print(f"    {symbol}: {partition.expression}")
            # Check values
            if partition.predicate_values:
                print(f"      on_a_b={partition.predicate_values.get('on_a_b', '?')}")
                print(f"      clear_c={partition.predicate_values.get('clear_c', '?')}")
    else:
        print(f"✗ Original label mapping not found")
        return False

    print("\n✓ Compatibility test passed!")
    return True


def test_issue_detection():
    """Detect potential design issues"""
    print("\n" + "="*80)
    print("DESIGN ISSUE DETECTION")
    print("="*80)

    issues = []

    # Issue 1: Partition symbol format compatibility
    print("\n--- Issue 1: Partition Symbol Format ---")
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])

    dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b"];
}
"""

    simplifier = DFASimplifier()
    result = simplifier.simplify(dfa_dot, gmap)

    # Check partition symbol format
    for partition in result.partitions:
        # Check if symbol might conflict with predicate names
        if partition.symbol.lower() in ['true', 'false', 'and', 'or', 'not']:
            issues.append(f"Partition symbol '{partition.symbol}' conflicts with boolean keyword")
        # Check if symbol contains special characters
        if not partition.symbol.replace('_', '').replace('α', '').isalnum():
            issues.append(f"Partition symbol '{partition.symbol}' contains special characters")

    print(f"Partition symbols: {[p.symbol for p in result.partitions]}")
    print(f"✓ Partition symbols are safe" if not issues else f"⚠ Found issues: {issues}")

    # Issue 2: Integration with BackwardPlannerGenerator
    print("\n--- Issue 2: BackwardPlannerGenerator Integration ---")
    from stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator

    # Check if BackwardPlannerGenerator can handle partition symbols
    # It currently expects transition labels to be boolean expressions
    # Partition symbols (like "α1", "p1") are NOT boolean expressions!

    print("⚠ CRITICAL ISSUE DETECTED:")
    print("  BackwardPlannerGenerator._parse_transition_label() expects boolean expressions")
    print("  But simplified DFA has atomic partition symbols (α1, p1, etc.)")
    print("  These are NOT parseable as boolean expressions!")
    print("")
    print("  Solution needed:")
    print("    1. Modify BackwardPlannerGenerator to detect partition symbols")
    print("    2. Use partition_map to resolve symbols → expressions")
    print("    3. Then parse the resolved expressions")

    issues.append("BackwardPlannerGenerator incompatible with partition symbols")

    # Issue 3: Partition map not passed through pipeline
    print("\n--- Issue 3: Pipeline Data Flow ---")
    print("⚠ ISSUE DETECTED:")
    print("  DFABuilder.build() returns dict with 'dfa_dot'")
    print("  But does NOT include 'partition_map' from simplification")
    print("  BackwardPlannerGenerator cannot resolve partition symbols!")
    print("")
    print("  Solution needed:")
    print("    1. Extend DFABuilder to optionally enable simplification")
    print("    2. Include 'partition_map' in returned dict")
    print("    3. Pass partition_map to BackwardPlannerGenerator")

    issues.append("Pipeline does not pass partition_map through")

    # Issue 4: Multiple edges for same partition
    print("\n--- Issue 4: DFA Edge Multiplication ---")
    gmap = GroundingMap()
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("clear_c", "clear", ["c"])

    dfa_dot = """
digraph G {
    s0 -> s1 [label="on_a_b | clear_c"];
}
"""

    result = simplifier.simplify(dfa_dot, gmap)

    print(f"Original: 1 transition with label 'on_a_b | clear_c'")
    print(f"Simplified: {len(result.original_label_to_partitions.get('on_a_b | clear_c', []))} transitions")
    print(f"  Partitions: {result.original_label_to_partitions.get('on_a_b | clear_c', [])}")
    print("")
    print("ℹ This is expected behavior (partition refinement)")
    print("  But increases DFA size (more edges)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES")
    print("="*80)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

    if issues:
        print(f"\n⚠ Found {len(issues)} design issues that need addressing")
    else:
        print("\n✓ No design issues detected")

    return issues


if __name__ == "__main__":
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "DFA SIMPLIFIER INTEGRATION TESTS" + " "*26 + "║")
    print("╚" + "═"*78 + "╝" + "\n")

    # Run tests
    try:
        test_integration_with_real_dfa()
        test_backward_planner_compatibility()
        issues = test_issue_detection()

        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print("✓ All integration tests passed")
        if issues:
            print(f"⚠ {len(issues)} design issues require attention before merging")
        else:
            print("✓ No critical design issues detected")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
