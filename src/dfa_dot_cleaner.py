"""
DFA DOT Cleaner

Extracts only essential DFA information from DOT format:
- States (nodes)
- Transitions (edges) with conditions
- Initial and accepting states

Removes Graphviz formatting directives (rankdir, size, shape, etc.)
"""

import re
from typing import Dict, List, Tuple


class DFAInfo:
    """Cleaned DFA information"""

    def __init__(self):
        self.init_state: str = ""
        self.states: List[str] = []
        self.accepting_states: List[str] = []
        self.transitions: List[Tuple[str, str, str]] = []  # (from, to, condition)

    def to_compact_string(self) -> str:
        """
        Convert to compact string representation

        Format:
        States: 1, 2, 3, 4
        Initial: 1
        Accepting: 4
        Transitions:
          1 -> 1 when [~on_b1_b2 & ~on_b2_b3]
          1 -> 2 when [on_b2_b3 & ~on_b1_b2]
          ...
        """
        lines = []

        lines.append(f"States: {', '.join(self.states)}")
        lines.append(f"Initial: {self.init_state}")
        lines.append(f"Accepting: {', '.join(self.accepting_states)}")

        if self.transitions:
            lines.append("Transitions:")
            for from_state, to_state, condition in self.transitions:
                lines.append(f"  {from_state} -> {to_state} when [{condition}]")

        return '\n'.join(lines)


def clean_dfa_dot(dfa_dot: str) -> DFAInfo:
    """
    Extract essential DFA information from DOT format

    Args:
        dfa_dot: Full DOT representation from ltlf2dfa

    Returns:
        DFAInfo with states, transitions, and conditions only
    """
    info = DFAInfo()

    # Extract accepting states
    # Format: node [shape = doublecircle]; 4; or node [shape = doublecircle]; 2; 4;
    accepting_pattern = r'node\s*\[shape\s*=\s*doublecircle\];\s*([0-9\s;]+)'
    accepting_match = re.search(accepting_pattern, dfa_dot)
    if accepting_match:
        accepting_str = accepting_match.group(1)
        # Extract all numbers
        info.accepting_states = re.findall(r'\d+', accepting_str)

    # Extract initial state
    # Format: init -> 1;
    init_pattern = r'init\s*->\s*(\d+)'
    init_match = re.search(init_pattern, dfa_dot)
    if init_match:
        info.init_state = init_match.group(1)

    # Extract transitions
    # Format: 1 -> 2 [label="condition"];
    transition_pattern = r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]+)"\]'
    transitions = re.findall(transition_pattern, dfa_dot)

    states_set = set()
    for from_state, to_state, condition in transitions:
        info.transitions.append((from_state, to_state, condition))
        states_set.add(from_state)
        states_set.add(to_state)

    # Sort states numerically
    info.states = sorted(states_set, key=lambda x: int(x))

    return info


def format_dfa_for_prompt(dfa_dot: str) -> str:
    """
    Format DFA DOT for inclusion in LLM prompts

    Extracts only essential information, removing Graphviz formatting

    Args:
        dfa_dot: Full DOT representation

    Returns:
        Compact string representation suitable for prompts
    """
    info = clean_dfa_dot(dfa_dot)
    return info.to_compact_string()


def test_dfa_cleaner():
    """Test the DFA cleaner"""
    # Sample DFA DOT from logs
    sample_dot = """digraph MONA_DFA {
 rankdir = LR;
 center = true;
 size = "7.5,10.5";
 edge [fontname = Courier];
 node [height = .5, width = .5];
 node [shape = doublecircle]; 4;
 node [shape = circle]; 1;
 init [shape = plaintext, label = ""];
 init -> 1;
 1 -> 1 [label="~on_b1_b2 & ~on_b2_b3"];
 1 -> 2 [label="on_b2_b3 & ~on_b1_b2"];
 1 -> 3 [label="on_b1_b2 & ~on_b2_b3"];
 1 -> 4 [label="on_b1_b2 & on_b2_b3"];
 2 -> 2 [label="~on_b1_b2"];
 2 -> 4 [label="on_b1_b2"];
 3 -> 3 [label="~on_b2_b3"];
 3 -> 4 [label="on_b2_b3"];
 4 -> 4 [label="true"];
}"""

    print("="*80)
    print("DFA DOT CLEANER TEST")
    print("="*80)
    print("\nOriginal DOT (truncated):")
    print(sample_dot[:200] + "...")
    print("\nCleaned DFA Info:")
    print("-"*80)
    cleaned = format_dfa_for_prompt(sample_dot)
    print(cleaned)
    print("="*80)


if __name__ == "__main__":
    test_dfa_cleaner()
