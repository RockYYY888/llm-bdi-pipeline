"""
Stage 1.5: LTLf to DFA Conversion

Converts LTLf specifications to Deterministic Finite Automata (DFA) using ltlf2dfa.

This stage bridges LTLf goal specifications and downstream processing by providing
a formal automaton representation that can be used for verification, planning, or
code generation.

Key Challenge: ltlf2dfa uses propositional variables, but our LTLf formulas use
predicates with arguments (e.g., on(a, b)). We handle this by:
1. Extracting all predicates from LTLf formulas
2. Creating propositional encodings (on(a,b) → on_a_b)
3. Converting formulas to propositional form
4. Generating DFA using ltlf2dfa
5. Maintaining mapping for downstream use

NOTE: Requires MONA (MONadic second-order logic Automata) to be installed.
      The setup_mona_path module automatically adds MONA to PATH.
"""

# Setup MONA path before importing ltlf2dfa
import sys
from pathlib import Path

# Add parent src directory to path for setup_mona_path import (only once)
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils.setup_mona_path import setup_mona
setup_mona()

import re
import subprocess
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Dict, List, Any, Tuple
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.ltlf2dfa import UNSAT_DOT
from ltlf2dfa.base import MonaProgram
from utils.symbol_normalizer import SymbolNormalizer


class PredicateToProposition:
    """
    Converts predicate-based LTLf formulas to propositional encoding

    Now uses SymbolNormalizer for consistent symbol handling across the pipeline.

    Examples:
        on(a, b) → on_a_b
        clear(a) → clear_a
        on(block-1, block-2) → on_blockhh1_blockhh2 (with hyphen encoding)
        handempty → handempty (already propositional)
    """

    def __init__(self, normalizer: SymbolNormalizer = None):
        """
        Initialize with optional normalizer instance

        Args:
            normalizer: SymbolNormalizer instance (creates new one if not provided)
        """
        self.normalizer = normalizer or SymbolNormalizer()

    def encode_predicate(self, predicate_str: str) -> str:
        """
        Encode a single predicate to propositional variable

        Args:
            predicate_str: e.g., "on(a, b)" or "clear(block-1)"

        Returns:
            Propositional variable: e.g., "on_a_b" or "clear_blockhh1"
        """
        # Parse predicate string using normalizer
        pred_name, args = self.normalizer.parse_predicate_string(predicate_str.strip())

        if not args:
            # Already propositional (e.g., "handempty", "a", "b")
            return predicate_str.strip()

        # Create propositional symbol using normalizer
        return self.normalizer.create_propositional_symbol(pred_name, args)

    def convert_formula(self, ltlf_formula_str: str) -> str:
        """
        Convert entire LTLf formula from predicate to propositional form

        Args:
            ltlf_formula_str: e.g., "F(on(a, b))" or "F(on(block-1, block-2)) & G(clear(c))"

        Returns:
            Propositional formula: e.g., "F(on_a_b)" or "F(on_blockhh1_blockhh2) & G(clear_c)"
        """
        # Use normalizer's formula normalization
        return self.normalizer.normalize_formula_string(ltlf_formula_str)

    def get_mapping(self) -> Dict[str, str]:
        """Get original → normalized mapping"""
        return self.normalizer.get_original_to_normalized_map()

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get normalized → original mapping"""
        return self.normalizer.get_normalized_to_original_map()


class LTLfToDFA:
    """
    Converts LTLf specifications to DFA using ltlf2dfa

    Pipeline: Predicate LTLf → Propositional LTLf → DFA (DOT format)
    """

    MONA_TIMEOUT_SECONDS = 300

    def __init__(self):
        self.ltlf_parser = LTLfParser()
        self.encoder = PredicateToProposition()

    def convert(self, ltl_spec: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Convert LTLf specification to DFA

        Args:
            ltl_spec: LTLSpecification object with formulas

        Returns:
            Tuple of (dfa_dot_string, metadata_dict)
            - dfa_dot_string: DFA in DOT format for visualization
            - metadata_dict: Contains mappings, original formulas, etc.
        """

        if hasattr(ltl_spec, "combined_formula_string"):
            original_formula = ltl_spec.combined_formula_string()
        elif hasattr(ltl_spec, 'formulas'):
            formula_strings = [f.to_string() for f in ltl_spec.formulas]
            if len(formula_strings) == 0:
                raise ValueError("No LTLf formulas provided")
            if len(formula_strings) == 1:
                original_formula = formula_strings[0]
            else:
                original_formula = " & ".join(f"({f})" for f in formula_strings)
        else:
            raise ValueError("ltl_spec must have 'formulas' attribute")

        # Convert to propositional encoding
        propositional_formula = self.encoder.convert_formula(original_formula)

        # Parse and convert to DFA
        try:
            formula_obj = self.ltlf_parser(propositional_formula)
            dfa_dot, parse_metadata = self._convert_formula_object_to_dot(formula_obj)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert LTLf to DFA.\n"
                f"Original formula: {original_formula}\n"
                f"Propositional formula: {propositional_formula}\n"
                f"Error: {str(e)}"
            ) from e

        # Prepare metadata
        metadata = {
            "original_formula": original_formula,
            "propositional_formula": propositional_formula,
            "predicate_to_prop_mapping": self.encoder.get_mapping(),
            "prop_to_predicate_mapping": self.encoder.get_reverse_mapping(),
            "num_states": int(
                parse_metadata.get("num_states") or self._count_dfa_states(dfa_dot)
            ),
            "num_transitions": int(
                parse_metadata.get("num_transitions")
                or self._count_dfa_transitions(dfa_dot)
            ),
            "alphabet": self._extract_alphabet(dfa_dot),
        }

        return dfa_dot, metadata

    def _convert_formula_object_to_dot(self, formula_obj: Any) -> Tuple[str, Dict[str, Any]]:
        """Convert an ltlf2dfa formula object to DOT without relying on sympy-heavy guard simplification."""
        if not hasattr(formula_obj, "to_mona") or not hasattr(formula_obj, "find_labels"):
            # Test doubles in the suite still expose the older zero-argument shape.
            return formula_obj.to_dfa(), {}
        mona_output = self._invoke_mona_directly(formula_obj)

        if mona_output is False:
            raise RuntimeError("MONA timed out while converting LTLf to DFA.")

        rendered_output = str(mona_output or "").strip()
        if not rendered_output:
            raise RuntimeError("MONA returned empty output while converting LTLf to DFA.")
        if rendered_output.lstrip().startswith("digraph"):
            return rendered_output, {}
        if "Formula is unsatisfiable" in rendered_output:
            return UNSAT_DOT, {
                "num_states": 1,
                "num_transitions": 1,
            }

        graph = self._parse_mona_output(rendered_output)
        return self._render_mona_graph_as_dot(graph), {
            "num_states": graph["num_states"],
            "num_transitions": graph["num_transitions"],
        }

    def _invoke_mona_directly(self, formula_obj: Any) -> str | bool:
        """Invoke MONA directly so we can strip oversized debug comments and tune timeout safely."""
        mona_program = self._render_mona_program(formula_obj)

        with TemporaryDirectory(prefix="ltlf2dfa_mona_") as temp_dir:
            program_path = Path(temp_dir) / "automa.mona"
            stdout_path = Path(temp_dir) / "stdout.txt"
            program_path.write_text(mona_program, encoding="utf-8")

            with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                try:
                    completed = subprocess.run(
                        ["mona", "-q", "-u", "-w", str(program_path)],
                        stdout=stdout_handle,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.MONA_TIMEOUT_SECONDS,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return False

            stdout_size = stdout_path.stat().st_size if stdout_path.exists() else 0
            stderr = str(completed.stderr or "").strip()
            if completed.returncode != 0:
                error_parts = [
                    f"MONA exited with code {completed.returncode}.",
                    f"stdout_size={stdout_size} bytes.",
                ]
                if stderr:
                    error_parts.append(f"stderr={stderr[:500]}")
                raise RuntimeError(" ".join(error_parts))
            if not stdout_size and stderr:
                raise RuntimeError(stderr[:500])
            return stdout_path.read_text(encoding="utf-8").strip()


    @staticmethod
    def _render_mona_program(formula_obj: Any) -> str:
        """Render a MONA program without embedding the full source formula as a leading comment."""
        program = MonaProgram(formula_obj).mona_program()
        lines = program.splitlines()
        if lines and lines[0].startswith("#"):
            return "\n".join(lines[1:]) + "\n"
        return program

    def _parse_mona_output(self, mona_output: str) -> Dict[str, Any]:
        """Parse raw MONA output into a grouped graph representation."""
        free_variables = self._extract_mona_free_variables(mona_output)
        accepting_states = self._extract_mona_accepting_states(mona_output)
        grouped_guards: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        init_targets: List[str] = []
        states: set[str] = set()

        for line in mona_output.splitlines():
            stripped = line.strip()
            if not stripped.startswith("State "):
                continue

            match = re.match(r"State\s+(\d+):\s*([01X]+)\s*->\s*state\s+(\d+)\s*$", stripped)
            if not match:
                continue

            source_state, guard_bits, target_state = match.groups()
            if source_state == "0":
                init_targets.append(target_state)
                continue

            grouped_guards[(source_state, target_state)].append(guard_bits)
            states.add(source_state)
            states.add(target_state)

        init_state = init_targets[0] if init_targets else (min(states) if states else "1")
        states.add(init_state)
        states.update(accepting_states)

        return {
            "free_variables": free_variables,
            "accepting_states": tuple(sorted(accepting_states, key=self._numeric_state_sort_key)),
            "init_state": init_state,
            "grouped_guards": {
                key: tuple(guards)
                for key, guards in grouped_guards.items()
            },
            "num_states": len(states),
            "num_transitions": len(grouped_guards),
        }

    @staticmethod
    def _extract_mona_free_variables(mona_output: str) -> Tuple[str, ...]:
        match = re.search(
            r"DFA for formula with free variables:\s*(.*?)\s*\n",
            mona_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ()
        raw_tokens = [
            token.strip().lower()
            for token in match.group(1).split()
            if token.strip()
        ]
        if len(raw_tokens) == 1 and raw_tokens[0] == "state":
            return ()
        return tuple(raw_tokens)

    @staticmethod
    def _extract_mona_accepting_states(mona_output: str) -> Tuple[str, ...]:
        match = re.search(
            r"Accepting states:\s*(.*?)\s*\n",
            mona_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ()
        return tuple(
            token.strip()
            for token in match.group(1).split()
            if token.strip()
        )

    def _render_mona_graph_as_dot(self, graph: Dict[str, Any]) -> str:
        """Render a grouped MONA graph into a compact DOT projection."""
        lines = [
            "digraph MONA_DFA {",
            ' rankdir = LR;',
            " center = true;",
            ' size = "7.5,10.5";',
            " edge [fontname = Courier];",
            " node [height = .5, width = .5];",
        ]

        accepting_states = tuple(graph.get("accepting_states", ()))
        if accepting_states:
            lines.append(
                f" node [shape = doublecircle]; {'; '.join(accepting_states)};",
            )
        else:
            lines.append(" node [shape = doublecircle];")

        init_state = str(graph.get("init_state") or "1")
        lines.append(f" node [shape = circle]; {init_state};")
        lines.append(' init [shape = plaintext, label = ""];')
        lines.append(f" init -> {init_state};")

        free_variables = tuple(graph.get("free_variables", ()))
        grouped_guards = graph.get("grouped_guards", {})
        for (source_state, target_state), guards in sorted(
            grouped_guards.items(),
            key=lambda item: (
                self._numeric_state_sort_key(item[0][0]),
                self._numeric_state_sort_key(item[0][1]),
            ),
        ):
            label = self._render_guard_group_label(tuple(guards), free_variables)
            lines.append(f' {source_state} -> {target_state} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def _render_guard_group_label(
        self,
        guards: Tuple[str, ...],
        free_variables: Tuple[str, ...],
    ) -> str:
        """Render a guard group compactly; large disjunctions stay opaque but parseable."""
        unique_guards = tuple(dict.fromkeys(str(guard).strip() for guard in guards if str(guard).strip()))
        if not unique_guards:
            return "false"
        if all(re.fullmatch(r"[01X]+", guard) is None for guard in unique_guards):
            return " | ".join(unique_guards)
        if len(unique_guards) == 1:
            return self._render_guard_cube(unique_guards[0], free_variables)
        if len(unique_guards) <= 8:
            rendered_cubes = [
                self._render_guard_cube(guard, free_variables)
                for guard in unique_guards
            ]
            return " | ".join(
                rendered if rendered == "true" else f"({rendered})"
                for rendered in rendered_cubes
            )
        return f"guard_group_{len(unique_guards)}"

    @staticmethod
    def _render_guard_cube(guard: str, free_variables: Tuple[str, ...]) -> str:
        literals: List[str] = []
        for index, value in enumerate(str(guard).strip()):
            if value == "X":
                continue
            if index >= len(free_variables):
                continue
            symbol = free_variables[index]
            literals.append(symbol if value == "1" else f"~{symbol}")
        return " & ".join(literals) if literals else "true"

    @staticmethod
    def _numeric_state_sort_key(state: str) -> Tuple[int, str]:
        token = str(state).strip()
        return (int(token), token) if token.isdigit() else (10**9, token)

    def _count_dfa_states(self, dfa_dot: str) -> int:
        """Count number of states in DFA from DOT representation"""
        states = set()
        for line in dfa_dot.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            grouped_match = re.search(r'node\s+\[.*?\];\s*([^;]+);', stripped)
            if grouped_match:
                tokens = re.findall(r"[A-Za-z0-9_]+", grouped_match.group(1))
                states.update(token for token in tokens if token != "init")
                continue
            single_match = re.search(r"([A-Za-z0-9_]+)\s*\[\s*shape\s*=\s*", stripped)
            if single_match:
                token = single_match.group(1)
                if token != "init":
                    states.add(token)
        return len(states)

    def _count_dfa_transitions(self, dfa_dot: str) -> int:
        """Count DFA transitions cheaply for large generic ltlf2dfa outputs."""
        total_edges = dfa_dot.count("->")
        init_edges = dfa_dot.count("init ->")
        return max(0, total_edges - init_edges)

    def _extract_alphabet(self, dfa_dot: str) -> List[str]:
        """Extract alphabet (propositional variables) from DFA"""
        # This would require parsing DOT more carefully
        # For now, return from our encoder mapping (normalized symbols)
        mapping = self.encoder.get_mapping()
        return list(mapping.values()) if mapping else []


def test_converter():
    """Test the LTLf to DFA converter"""

    # Example LTL specification
    class ExampleFormula:
        def __init__(self, formula_str):
            self.formula_str = formula_str

        def to_string(self):
            return self.formula_str

    class ExampleLTLSpec:
        def __init__(self, formulas):
            self.formulas = [ExampleFormula(f) for f in formulas]

    print("="*80)
    print("LTLf TO DFA CONVERTER TEST")
    print("="*80)
    print()

    converter = LTLfToDFA()

    # Test cases
    test_cases = [
        (["F(on(a, b))"], "Single goal: Eventually on(a, b)"),
        (["F(on(a, b))", "F(clear(a))"], "Multiple goals: Eventually on(a,b) and clear(a)"),
        (["G(handempty)"], "Global constraint: Always handempty"),
        (["F(on(a, b))", "G(clear(c))"], "Mixed: Eventually on(a,b), always clear(c)"),
    ]

    for formulas, description in test_cases:
        print(f"Test: {description}")
        print(f"Input formulas: {formulas}")

        spec = ExampleLTLSpec(formulas)

        try:
            dfa_dot, metadata = converter.convert(spec)

            print(f"✓ Conversion successful")
            print(f"  Original: {metadata['original_formula']}")
            print(f"  Propositional: {metadata['propositional_formula']}")
            print(f"  Mappings: {metadata['predicate_to_prop_mapping']}")
            print(f"\n  COMPLETE DFA (DOT format):")
            print("  " + "~" * 76)
            for line in dfa_dot.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 76)
            print()

        except Exception as e:
            print(f"✗ Failed: {e}")
            print()


if __name__ == "__main__":
    test_converter()
