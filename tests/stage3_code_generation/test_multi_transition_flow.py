"""
Multi-Transition Flow Visualization

This test demonstrates how the system handles multiple DFA transitions:
1. Parse each transition label as a goal
2. Run backward planning for each goal separately
3. Merge all generated AgentSpeak code sections

Example DFA with 2 transitions:
  state0 --[on(a,b)]-> state1 --[clear(c)]-> state2
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def test_two_transition_dfa():
    """
    Test DFA with 2 sequential transitions

    DFA:
      state0 --[on_a_b]-> state1 --[clear_a]-> state2 (accepting)

    This means:
      1. First achieve on(a, b)
      2. Then achieve clear(a)
    """
    print("="*80)
    print("MULTI-TRANSITION FLOW TEST")
    print("="*80)
    print()

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map for BOTH goals
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_a", "clear", ["a"])

    # DFA with 2 transitions
    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1"];
    state2 [label="2", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
    state1 -> state2 [label="clear_a"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b) & F(clear(a)))",
        "dfa_dot": dfa_dot,
        "num_states": 3,
        "num_transitions": 2
    }

    ltl_dict = {
        "objects": ["a", "b"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("DFA Structure:")
    print("  state0 --[on_a_b]-> state1 --[clear_a]-> state2")
    print()
    print("Expected behavior:")
    print("  1. Generate plans for achieving on(a, b) from ANY state")
    print("  2. Generate plans for achieving clear(a) from ANY state")
    print("  3. Merge both sets of plans into one AgentSpeak file")
    print()

    print("="*80)
    print("STEP-BY-STEP EXECUTION")
    print("="*80)
    print()

    generator = BackwardPlannerGenerator(domain, grounding_map)

    # Generate code
    asl_code = generator.generate(ltl_dict, dfa_result)

    print("\n" + "="*80)
    print("ANALYSIS OF GENERATED CODE")
    print("="*80)
    print()

    # Check structure
    sections = asl_code.split("/* ========== Next Goal ========== */")

    print(f"Total code sections: {len(sections)}")
    print()

    # Analyze each section
    for i, section in enumerate(sections):
        print(f"--- Section {i+1} ---")

        # Count plans in this section
        plan_count = section.count("+!")
        print(f"  Plans (starting with +!): {plan_count}")

        # Check which goals are in this section
        has_on_ab = "on(a, b)" in section or "on_a_b" in section
        has_clear_a = "clear(a)" in section or "clear_a" in section

        if has_on_ab:
            print(f"  ✓ Contains plans for: on(a, b)")
        if has_clear_a:
            print(f"  ✓ Contains plans for: clear(a)")

        # Show first few lines of section
        lines = section.strip().split('\n')[:5]
        if lines and lines[0]:
            print(f"  First few lines:")
            for line in lines:
                if line.strip():
                    print(f"    {line[:80]}")
        print()

    # Check for potential issues
    print("="*80)
    print("POTENTIAL ISSUES TO CHECK")
    print("="*80)
    print()

    issues = []

    # Issue 1: Are plans independent or combined?
    if "clear(a)" in sections[0] and "on(a, b)" in sections[0]:
        issues.append("⚠️  Both goals appear in first section - may be combined incorrectly")
    else:
        print("✓ Goals appear in separate sections (as expected)")

    # Issue 2: Are there duplicate plans?
    all_plan_signatures = []
    import re
    for match in re.finditer(r'\+!(\w+)\s*(?:\([^)]*\))?\s*:', asl_code):
        plan_sig = match.group(0)
        all_plan_signatures.append(plan_sig)

    duplicate_count = len(all_plan_signatures) - len(set(all_plan_signatures))
    if duplicate_count > 0:
        issues.append(f"⚠️  {duplicate_count} duplicate plan signatures found")
    else:
        print(f"✓ No duplicate plan signatures ({len(all_plan_signatures)} unique plans)")

    # Issue 3: Are initial states covered?
    if "ontable(a)" in asl_code and "ontable(b)" in asl_code:
        print("✓ Initial state scenarios are covered")
    else:
        issues.append("⚠️  Initial state (blocks on table) may not be covered")

    # Issue 4: Size check
    print(f"✓ Total code size: {len(asl_code):,} characters")

    if len(asl_code) > 100000:
        issues.append(f"⚠️  Code is very large ({len(asl_code):,} chars)")

    print()

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No issues found!")

    print()

    # Summary
    print("="*80)
    print("SUMMARY: How Multi-Transition DFA is Handled")
    print("="*80)
    print()
    print("Process:")
    print("  1. Parse DFA → extract transitions")
    print(f"     Found {len(sections)} transition(s)")
    print()
    print("  2. For EACH transition:")
    print("     a. Parse label as goal (e.g., 'on_a_b' → on(a, b))")
    print("     b. Run backward planning from that goal")
    print("     c. Generate AgentSpeak plans for all reachable states")
    print()
    print("  3. Merge all code sections:")
    print("     - Join with separator: /* ========== Next Goal ========== */")
    print("     - Each section is INDEPENDENT")
    print("     - No deduplication between sections")
    print()
    print("Key Design Decision:")
    print("  ✓ Each transition is planned SEPARATELY")
    print("  ✓ Plans don't depend on each other")
    print("  ✓ Agent can achieve ANY goal from ANY reachable state")
    print()
    print("Implications:")
    print("  ✓ Pros: Simple, modular, correct")
    print("  ⚠️  Cons: May generate redundant plans across sections")
    print("  ⚠️  Cons: Code size grows linearly with # transitions")
    print()


def test_conjunctive_transition():
    """
    Test DFA with conjunctive transition label

    DFA:
      state0 --[on_a_b & clear_c]-> state1 (accepting)

    This means: achieve BOTH on(a,b) AND clear(c) simultaneously
    """
    print("\n\n" + "="*80)
    print("CONJUNCTIVE TRANSITION TEST")
    print("="*80)
    print()

    # Load domain
    domain_path = project_root / "src" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    # Create grounding map
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])
    grounding_map.add_atom("clear_c", "clear", ["c"])

    # DFA with conjunctive label
    dfa_dot = """
digraph {
    rankdir=LR;
    node [shape=circle];
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="on_a_b & clear_c"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b) & clear(c))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    ltl_dict = {
        "objects": ["a", "b", "c"],
        "formulas_string": [dfa_result["formula"]],
        "grounding_map": grounding_map
    }

    print("DFA Structure:")
    print("  state0 --[on_a_b & clear_c]-> state1")
    print()
    print("This is a CONJUNCTIVE goal: both predicates must be true")
    print()

    generator = BackwardPlannerGenerator(domain, grounding_map)
    asl_code = generator.generate(ltl_dict, dfa_result)

    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    sections = asl_code.split("/* ========== Next Goal ========== */")
    print(f"Code sections: {len(sections)}")
    print()

    if len(sections) == 1:
        print("✓ Conjunctive goal treated as SINGLE planning problem")
        print("  Goal state: {on(a, b), clear(c)}")
        print("  Plans generated for achieving both predicates together")
    else:
        print("⚠️  Conjunctive goal split into multiple sections")
        print("  This may not be correct for conjunctive goals")

    print()
    print(f"Total code size: {len(asl_code):,} characters")
    print()


if __name__ == "__main__":
    # Test 1: Sequential transitions
    test_two_transition_dfa()

    # Test 2: Conjunctive transition
    test_conjunctive_transition()

    print("\n" + "="*80)
    print("FINAL INSIGHTS")
    print("="*80)
    print()
    print("How the system handles DFA transitions:")
    print()
    print("1. SEQUENTIAL transitions (state0→state1→state2):")
    print("   - Each transition planned independently")
    print("   - N transitions → N code sections")
    print("   - Plans merged with separators")
    print()
    print("2. CONJUNCTIVE labels (on_a_b & clear_c):")
    print("   - Parsed as single goal with multiple predicates")
    print("   - Backward planning from joint goal state")
    print("   - 1 transition → 1 code section")
    print()
    print("3. DISJUNCTIVE labels (on_a_b | clear_c) - if supported:")
    print("   - Would be converted to DNF")
    print("   - Each disjunct planned separately")
    print("   - 1 transition → M code sections (M = # disjuncts)")
    print()
