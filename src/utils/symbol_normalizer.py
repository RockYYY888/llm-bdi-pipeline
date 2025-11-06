"""
Symbol Normalizer Module

Handles normalization of symbols for LTLf formulas to ensure compatibility with ltlf2dfa.
This module centralizes all symbol transformation logic including special character handling.

Key transformations:
1. Hyphen (-) handling: Replace with "hh" to avoid parsing issues
2. Predicate to propositional conversion: on(a, b) → on_a_b
3. Reverse transformations for human-readable output

Design rationale:
- ltlf2dfa has strict requirements for propositional symbols
- Hyphens in object names (e.g., "block-1", "rover-a") can cause parsing errors
- We normalize during LTLf→DFA conversion and denormalize for output
"""

import re
from typing import List, Tuple, Dict


class SymbolNormalizer:
    """
    Centralizes all symbol normalization logic for LTLf processing

    Handles:
    - Hyphen replacement/restoration (- ↔ hh)
    - Predicate to propositional symbol conversion
    - Special character escaping for ltlf2dfa compatibility
    """

    # Constants for hyphen encoding
    HYPHEN_REPLACEMENT = "hh"
    HYPHEN_CHAR = "-"

    # Regex patterns
    PREDICATE_PATTERN = re.compile(r'([a-z_][a-z0-9_]*)\(([^()]+)\)')

    def __init__(self):
        """Initialize normalizer with empty mapping tables"""
        self.normalized_to_original: Dict[str, str] = {}
        self.original_to_normalized: Dict[str, str] = {}

    # ========== HYPHEN HANDLING ==========

    def encode_hyphens(self, text: str) -> str:
        """
        Replace all hyphens with 'hh' encoding

        Args:
            text: Input string potentially containing hyphens

        Returns:
            String with hyphens replaced by 'hh'

        Examples:
            "block-1" → "blockhh1"
            "rover-a" → "roverhha"
            "clear(block-1)" → "clear(blockhh1)"
        """
        if self.HYPHEN_CHAR not in text:
            return text

        normalized = text.replace(self.HYPHEN_CHAR, self.HYPHEN_REPLACEMENT)

        # Store mapping for reverse conversion
        if text not in self.original_to_normalized:
            self.original_to_normalized[text] = normalized
            self.normalized_to_original[normalized] = text

        return normalized

    def decode_hyphens(self, text: str) -> str:
        """
        Restore hyphens from 'hh' encoding

        Args:
            text: Input string with 'hh' encoding

        Returns:
            String with 'hh' replaced by hyphens

        Examples:
            "blockhh1" → "block-1"
            "roverhha" → "rover-a"
            "clear_blockhh1" → "clear_block-1"
        """
        if self.HYPHEN_REPLACEMENT not in text:
            return text

        # First check if we have a stored mapping
        if text in self.normalized_to_original:
            return self.normalized_to_original[text]

        # Otherwise, do direct replacement
        return text.replace(self.HYPHEN_REPLACEMENT, self.HYPHEN_CHAR)

    # ========== PREDICATE NORMALIZATION ==========

    def normalize_predicate_args(self, predicate: str, args: List[str]) -> Tuple[str, List[str]]:
        """
        Normalize predicate name and arguments (handle hyphens)

        Args:
            predicate: Predicate name (e.g., "on", "clear")
            args: List of arguments (e.g., ["block-1", "block-2"])

        Returns:
            Tuple of (normalized_predicate, normalized_args)

        Examples:
            ("on", ["block-1", "table"]) → ("on", ["blockhh1", "table"])
            ("clear", ["rover-a"]) → ("clear", ["roverhha"])
        """
        normalized_predicate = self.encode_hyphens(predicate)
        normalized_args = [self.encode_hyphens(arg) for arg in args]

        return normalized_predicate, normalized_args

    def create_propositional_symbol(self, predicate: str, args: List[str]) -> str:
        """
        Create propositional symbol from predicate and arguments

        Handles hyphen encoding automatically.

        Args:
            predicate: Predicate name
            args: List of arguments (may contain hyphens)

        Returns:
            Propositional symbol with normalized format

        Examples:
            ("on", ["a", "b"]) → "on_a_b"
            ("on", ["block-1", "block-2"]) → "on_blockhh1_blockhh2"
            ("clear", ["rover-a"]) → "clear_roverhha"
            ("handempty", []) → "handempty"
        """
        # Normalize predicate and args (encode hyphens)
        norm_pred, norm_args = self.normalize_predicate_args(predicate, args)

        # Nullary predicate (no arguments)
        if not norm_args:
            return norm_pred.lower()

        # Join with underscores, all lowercase
        symbol = f"{norm_pred.lower()}_{'_'.join(arg.lower() for arg in norm_args)}"

        # Store reverse mapping for the entire symbol
        original_symbol = self._create_original_symbol(predicate, args)
        self.normalized_to_original[symbol] = original_symbol
        self.original_to_normalized[original_symbol] = symbol

        return symbol

    def _create_original_symbol(self, predicate: str, args: List[str]) -> str:
        """Create original symbol (with hyphens) for reverse mapping"""
        if not args:
            return predicate.lower()
        return f"{predicate.lower()}_{'_'.join(arg.lower() for arg in args)}"

    def restore_symbol_hyphens(self, normalized_symbol: str) -> str:
        """
        Restore hyphens in a propositional symbol

        Args:
            normalized_symbol: Symbol with 'hh' encoding

        Returns:
            Symbol with hyphens restored

        Examples:
            "on_blockhh1_blockhh2" → "on_block-1_block-2"
            "clear_roverhha" → "clear_rover-a"
        """
        # Check mapping first
        if normalized_symbol in self.normalized_to_original:
            return self.normalized_to_original[normalized_symbol]

        # Otherwise decode directly
        return self.decode_hyphens(normalized_symbol)

    # ========== FULL FORMULA CONVERSION ==========

    def normalize_formula_string(self, formula_str: str) -> str:
        """
        Normalize an entire LTLf formula string

        Converts predicates to propositional symbols and encodes hyphens.
        Reserved LTL operators (F, G, X, WX, U, R, W, M) are preserved.

        Args:
            formula_str: LTLf formula with predicates, e.g., "F(on(block-1, block-2))"

        Returns:
            Normalized formula, e.g., "F(on_blockhh1_blockhh2)"

        Examples:
            "F(on(a, b))" → "F(on_a_b)"
            "F(on(block-1, table))" → "F(on_blockhh1_table)"
            "G(clear(rover-a))" → "G(clear_roverhha)"
        """
        # Reserved LTL operators that should NOT be replaced
        ltl_operators = {'F', 'G', 'X', 'WX', 'U', 'R', 'W', 'M'}

        def replacer(match):
            full_match = match.group(0)  # e.g., "on(block-1, block-2)"
            pred_name = match.group(1)    # e.g., "on"
            args_str = match.group(2)     # e.g., "block-1, block-2"

            # Skip if it's an LTL operator
            if pred_name in ltl_operators:
                return full_match

            # Parse arguments
            args = [arg.strip() for arg in args_str.split(',')]

            # Create normalized propositional symbol
            return self.create_propositional_symbol(pred_name, args)

        # Apply replacements iteratively to handle nested structures
        prev_converted = formula_str
        max_iterations = 10

        for _ in range(max_iterations):
            converted = self.PREDICATE_PATTERN.sub(replacer, prev_converted)

            if converted == prev_converted:
                # No more changes
                break

            prev_converted = converted
        else:
            raise RuntimeError(
                f"Failed to normalize formula after {max_iterations} iterations. "
                f"Formula: {formula_str}"
            )

        return converted

    def denormalize_formula_string(self, normalized_formula: str) -> str:
        """
        Restore hyphens in a normalized LTLf formula string

        Args:
            normalized_formula: Formula with 'hh' encoding

        Returns:
            Formula with hyphens restored

        Examples:
            "F(on_blockhh1_blockhh2)" → "F(on_block-1_block-2)"
            "G(clear_roverhha)" → "G(clear_rover-a)"
        """
        return self.decode_hyphens(normalized_formula)

    # ========== PREDICATE STRING PARSING ==========

    def parse_predicate_string(self, predicate_str: str) -> Tuple[str, List[str]]:
        """
        Parse a predicate string into predicate name and arguments

        Args:
            predicate_str: e.g., "on(a, b)" or "clear(block-1)"

        Returns:
            Tuple of (predicate_name, args_list)

        Examples:
            "on(a, b)" → ("on", ["a", "b"])
            "clear(block-1)" → ("clear", ["block-1"])
            "handempty" → ("handempty", [])
        """
        match = self.PREDICATE_PATTERN.match(predicate_str.strip())

        if not match:
            # No arguments (nullary predicate or propositional constant)
            return predicate_str.strip(), []

        pred_name = match.group(1)
        args_str = match.group(2)
        args = [arg.strip() for arg in args_str.split(',')]

        return pred_name, args

    # ========== ANTI-GROUNDING (PROPOSITIONAL → PARAMETERIZED) ==========

    def symbol_to_parameterized(self, propositional_symbol: str) -> str:
        """
        Convert propositional symbol to parameterized predicate format

        This is the anti-grounding operation that converts flattened symbols
        back to human-readable parameterized predicates.

        Args:
            propositional_symbol: e.g., "on_a_b", "clear_c", "handempty"

        Returns:
            Parameterized predicate format: e.g., "on(a, b)", "clear(c)", "handempty"

        Examples:
            "on_a_b" → "on(a, b)"
            "on_blockhh1_blockhh2" → "on(block-1, block-2)"
            "clear_c" → "clear(c)"
            "handempty" → "handempty"
            "at_roverhh1_locationhhbase" → "at(rover-1, location-base)"
        """
        # First restore hyphens if encoded
        restored = self.restore_symbol_hyphens(propositional_symbol)

        # Parse the symbol: predicate_arg1_arg2_...
        parts = restored.split('_')

        if len(parts) == 1:
            # Nullary predicate (no arguments)
            return parts[0]

        # First part is predicate, rest are arguments
        predicate = parts[0]
        args = parts[1:]

        # Format as predicate(arg1, arg2, ...)
        return f"{predicate}({', '.join(args)})"

    def format_grounding_map_for_antigrounding(self, grounding_map: Dict) -> str:
        """
        Format grounding map for anti-grounding reference in prompts

        Creates a human-readable conversion table showing how to convert
        propositional symbols back to parameterized predicates.

        Args:
            grounding_map: Dict with 'atoms' key containing symbol mappings

        Returns:
            Formatted string with conversion table grouped by predicate

        Example output:
            ```
            **Grounding Map (Anti-Grounding Reference)**:
            Use this to convert propositional symbols to parameterized predicates:

              clear:
                clear_b1 → clear(b1)
                clear_b3 → clear(b3)

              on:
                on_b1_b2 → on(b1, b2)
                on_b2_b5 → on(b2, b5)
            ```
        """
        if not grounding_map or 'atoms' not in grounding_map:
            return ""

        atoms = grounding_map.get('atoms', {})
        if not atoms:
            return ""

        info = "\n**Grounding Map (Anti-Grounding Reference)**:\n"
        info += "Use this to convert propositional symbols to parameterized predicates:\n\n"

        # Group by predicate for better readability
        predicate_groups = {}
        for symbol, atom_data in atoms.items():
            predicate = atom_data.get('predicate', '')
            if predicate not in predicate_groups:
                predicate_groups[predicate] = []
            predicate_groups[predicate].append((symbol, atom_data))

        for predicate, group in sorted(predicate_groups.items()):
            info += f"  {predicate}:\n"
            for symbol, atom_data in sorted(group):
                args = atom_data.get('args', [])
                if args:
                    param_form = f"{predicate}({', '.join(args)})"
                else:
                    param_form = predicate
                info += f"    {symbol} → {param_form}\n"
            info += "\n"

        return info

    # ========== MAPPING UTILITIES ==========

    def get_normalized_to_original_map(self) -> Dict[str, str]:
        """Get mapping from normalized symbols to original symbols (with hyphens)"""
        return dict(self.normalized_to_original)

    def get_original_to_normalized_map(self) -> Dict[str, str]:
        """Get mapping from original symbols to normalized symbols"""
        return dict(self.original_to_normalized)

    def clear_mappings(self):
        """Clear all stored mappings (useful for fresh conversions)"""
        self.normalized_to_original.clear()
        self.original_to_normalized.clear()


def test_symbol_normalizer():
    """Test the SymbolNormalizer class"""
    print("=" * 80)
    print("SYMBOL NORMALIZER TEST")
    print("=" * 80)
    print()

    normalizer = SymbolNormalizer()

    # Test 1: Hyphen encoding/decoding
    print("Test 1: Hyphen Encoding/Decoding")
    print("-" * 40)
    test_cases_hyphen = [
        "block-1",
        "rover-a",
        "location-base",
        "no-hyphens-here",
    ]
    for text in test_cases_hyphen:
        encoded = normalizer.encode_hyphens(text)
        decoded = normalizer.decode_hyphens(encoded)
        print(f"  {text:20} → {encoded:20} → {decoded:20} ✓" if decoded == text else f"  FAIL: {text}")
    print()

    # Test 2: Propositional symbol creation
    print("Test 2: Propositional Symbol Creation")
    print("-" * 40)
    test_cases_symbols = [
        ("on", ["a", "b"], "on_a_b"),
        ("on", ["block-1", "block-2"], "on_blockhh1_blockhh2"),
        ("clear", ["rover-a"], "clear_roverhha"),
        ("handempty", [], "handempty"),
        ("at", ["rover-1", "location-base"], "at_roverhh1_locationhhbase"),
    ]
    for predicate, args, expected in test_cases_symbols:
        result = normalizer.create_propositional_symbol(predicate, args)
        status = "✓" if result == expected else f"✗ (got: {result})"
        print(f"  {predicate}({', '.join(args):30}) → {result:30} {status}")
    print()

    # Test 3: Formula normalization
    print("Test 3: Formula Normalization")
    print("-" * 40)
    test_cases_formulas = [
        ("F(on(a, b))", "F(on_a_b)"),
        ("F(on(block-1, block-2))", "F(on_blockhh1_blockhh2)"),
        ("G(clear(rover-a))", "G(clear_roverhha)"),
        ("F(on(a, b)) & G(clear(c))", "F(on_a_b) & G(clear_c)"),
        ("(on(block-1, table) U clear(block-2))", "(on_blockhh1_table U clear_blockhh2)"),
    ]
    for original, expected in test_cases_formulas:
        result = normalizer.normalize_formula_string(original)
        status = "✓" if result == expected else f"✗ (got: {result})"
        print(f"  {original:45} → {result:45} {status}")
    print()

    # Test 4: Symbol restoration
    print("Test 4: Symbol Restoration")
    print("-" * 40)
    test_cases_restore = [
        ("on_blockhh1_blockhh2", "on_block-1_block-2"),
        ("clear_roverhha", "clear_rover-a"),
        ("at_roverhh1_locationhhbase", "at_rover-1_location-base"),
    ]
    for normalized, expected in test_cases_restore:
        result = normalizer.restore_symbol_hyphens(normalized)
        status = "✓" if result == expected else f"✗ (got: {result})"
        print(f"  {normalized:35} → {result:35} {status}")
    print()


if __name__ == "__main__":
    test_symbol_normalizer()
