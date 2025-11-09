"""
Visualize Multi-Transition Processing Step-by-Step
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def visualize_multi_transition_processing():
    print("=" * 80)
    print("MULTI-TRANSITION PROCESSING - STEP BY STEP VISUALIZATION")
    print("=" * 80)
    print()

    # Example DFA
    dfa_dot = """
digraph {
    state0 -> state1 [label="on_a_b"];
    state1 -> state2 [label="clear_a"];
}
"""

    print("DFA (DOT format):")
    print(dfa_dot)
    print()

    # Step 1: Parse DFA
    print("=" * 80)
    print("STEP 1: Parse DFA → Extract Transitions")
    print("=" * 80)
    print()

    transitions = [
        ("state0", "state1", "on_a_b"),
        ("state1", "state2", "clear_a")
    ]

    for i, (from_state, to_state, label) in enumerate(transitions, 1):
        print(f"Transition {i}: {from_state} --[{label}]-> {to_state}")
    print()

    # Step 2: Parse labels
    print("=" * 80)
    print("STEP 2: Parse Labels → Goal Predicates")
    print("=" * 80)
    print()

    # Grounding map (pre-defined)
    grounding_map = {
        "on_a_b": ("on", ["a", "b"]),
        "clear_a": ("clear", ["a"])
    }

    print("Grounding Map:")
    for atom_id, (predicate, args) in grounding_map.items():
        print(f"  {atom_id} → {predicate}({', '.join(args)})")
    print()

    print("Label → Goal Predicates:")
    for i, (from_state, to_state, label) in enumerate(transitions, 1):
        predicate, args = grounding_map[label]
        print(f"  Transition {i}: '{label}' → [{predicate}({', '.join(args)})]")
    print()

    # Step 3: Backward planning for each
    print("=" * 80)
    print("STEP 3: Run Backward Planning for Each Goal")
    print("=" * 80)
    print()

    for i, (from_state, to_state, label) in enumerate(transitions, 1):
        predicate, args = grounding_map[label]
        goal = f"{predicate}({', '.join(args)})"

        print(f"Transition {i}: Goal = {goal}")
        print(f"  1. Create ForwardStatePlanner(domain, objects)")
        print(f"  2. Run planner.explore_from_goal([{goal}])")
        print(f"  3. Result: StateGraph with ~1000+ states")
        print(f"  4. Generate AgentSpeak code for this goal")
        print(f"  5. Add to code_sections list")
        print()

    # Step 4: Merge code
    print("=" * 80)
    print("STEP 4: Merge All Code Sections")
    print("=" * 80)
    print()

    print("Structure of final code:")
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│ Main Header                                  │")
    print("│  - LTL formula                               │")
    print("│  - Objects                                   │")
    print("│  - DFA statistics                            │")
    print("└─────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│ Code Section 1: Goal on(a, b)               │")
    print("│                                              │")
    print("│ /* Initial Beliefs */                        │")
    print("│ ontable(a). clear(a). ...                    │")
    print("│                                              │")
    print("│ /* Action Plans */                           │")
    print("│ +!pick_up(B1, B2) : ... <- ...              │")
    print("│ +!put_on_block(B1, B2) : ... <- ...         │")
    print("│ ...                                          │")
    print("│                                              │")
    print("│ /* Goal Achievement Plans */                 │")
    print("│ +!on(a, b) : holding(a) & clear(b) <- ...   │")
    print("│ +!on(a, b) : handempty & ontable(a) <- ...  │")
    print("│ ... (26 plans total)                         │")
    print("└─────────────────────────────────────────────┘")
    print()
    print("/* ========== Next Goal ========== */")
    print()
    print("┌─────────────────────────────────────────────┐")
    print("│ Code Section 2: Goal clear(a)               │")
    print("│                                              │")
    print("│ /* Initial Beliefs */                        │")
    print("│ ontable(a). clear(a). ...                    │")
    print("│                                              │")
    print("│ /* Action Plans */                           │")
    print("│ +!pick_up(B1, B2) : ... <- ...              │")
    print("│ +!put_down(B) : ... <- ...                  │")
    print("│ ...                                          │")
    print("│                                              │")
    print("│ /* Goal Achievement Plans */                 │")
    print("│ +!clear(a) : ontable(a) & clear(a) <- ...   │")
    print("│ +!clear(a) : holding(a) <- ...               │")
    print("│ ... (different plans for clear(a))           │")
    print("└─────────────────────────────────────────────┘")
    print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    print("1. Each transition is INDEPENDENT:")
    print("   - Different goal predicates")
    print("   - Different state spaces")
    print("   - Different plans generated")
    print()

    print("2. Label parsing uses GroundingMap:")
    print("   - 'on_a_b' is an atom ID (from Stage 1)")
    print("   - GroundingMap stores: atom_id → (predicate, args)")
    print("   - Supports complex labels: 'on_a_b & clear_c'")
    print()

    print("3. Code merging is simple concatenation:")
    print("   - Separator: '/* ========== Next Goal ========== */'")
    print("   - Each section is self-contained")
    print("   - Agent runtime chooses appropriate plans based on current goal")
    print()

    print("4. Why NOT merge state spaces?")
    print("   - on(a, b) and clear(a) have DIFFERENT reachable states")
    print("   - Merging would lose goal-specific information")
    print("   - Each goal needs its own complete plan library")
    print()


if __name__ == "__main__":
    visualize_multi_transition_processing()
