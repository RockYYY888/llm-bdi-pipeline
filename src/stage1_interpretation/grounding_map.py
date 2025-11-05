"""
Grounding Map for Propositional LTLf

Manages the mapping between propositional symbols (e.g., on_a_b) and
their original parameterized predicates (e.g., on(a, b)).

This is essential because LTLf only supports propositional atoms, not
parameterized predicates. The grounding map serves as a bridge between:
- LTLf formulas (using propositional symbols)
- Domain knowledge (predicates and objects from PDDL)
- AgentSpeak code generation (needs human-readable predicates)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import sys
from pathlib import Path

# Add parent directory to path for utils import
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from utils.symbol_normalizer import SymbolNormalizer


@dataclass
class GroundedAtom:
    """
    Represents a grounded propositional atom

    Example:
        symbol: "on_a_b"
        predicate: "on"
        args: ["a", "b"]
    """
    symbol: str  # Propositional symbol (e.g., "on_a_b")
    predicate: str  # Original predicate name (e.g., "on")
    args: List[str]  # Arguments (e.g., ["a", "b"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "symbol": self.symbol,
            "predicate": self.predicate,
            "args": self.args
        }

    def to_pddl_format(self) -> str:
        """Convert to PDDL format: (on a b)"""
        if not self.args:
            return self.predicate
        return f"({self.predicate} {' '.join(self.args)})"

    def to_readable_format(self) -> str:
        """Convert to human-readable format: on(a, b)"""
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"


class GroundingMap:
    """
    Central grounding map that maintains all propositional atoms
    and their mappings to domain predicates/objects.

    Schema:
    {
      "atoms": {
        "on_a_b": {"predicate":"on", "args":["a","b"]},
        "clear_a": {"predicate":"clear", "args":["a"]},
        "handempty": {"predicate":"handempty", "args":[]}
      },
      "predicates": {
        "on": {"arity":2},
        "clear": {"arity":1},
        "handempty": {"arity":0}
      },
      "objects": ["a","b","c"]
    }
    """

    def __init__(self, normalizer: SymbolNormalizer = None):
        self.atoms: Dict[str, GroundedAtom] = {}
        self.predicates: Dict[str, int] = {}  # predicate -> arity
        self.objects: List[str] = []
        self.normalizer = normalizer or SymbolNormalizer()

    def add_atom(self, symbol: str, predicate: str, args: List[str]) -> GroundedAtom:
        """
        Add a grounded atom to the map

        Args:
            symbol: Propositional symbol (e.g., "on_a_b")
            predicate: Predicate name (e.g., "on")
            args: List of arguments (e.g., ["a", "b"])

        Returns:
            The GroundedAtom object
        """
        atom = GroundedAtom(symbol=symbol, predicate=predicate, args=args)
        self.atoms[symbol] = atom

        # Update predicates registry
        if predicate not in self.predicates:
            self.predicates[predicate] = len(args)

        # Update objects registry
        for arg in args:
            if arg not in self.objects:
                self.objects.append(arg)

        return atom

    def get_atom(self, symbol: str) -> GroundedAtom:
        """Get grounded atom by symbol"""
        return self.atoms.get(symbol)

    def get_all_atoms(self) -> List[GroundedAtom]:
        """Get all grounded atoms"""
        return list(self.atoms.values())

    def validate_atom(self, symbol: str, domain_predicates: List[str],
                     domain_objects: List[str]) -> List[str]:
        """
        Validate that an atom's predicate and objects exist in domain

        Args:
            symbol: Symbol to validate
            domain_predicates: List of valid predicates from domain
            domain_objects: List of valid objects from domain

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        atom = self.atoms.get(symbol)

        if not atom:
            errors.append(f"Atom '{symbol}' not found in grounding map")
            return errors

        # Check predicate exists (skip propositional constants)
        if atom.predicate not in ['true', 'false'] and atom.predicate not in domain_predicates:
            errors.append(f"Unknown predicate '{atom.predicate}' in atom '{symbol}'")

        # Check all arguments are valid objects
        for arg in atom.args:
            if arg not in domain_objects:
                errors.append(f"Unknown object '{arg}' in atom '{symbol}'")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "atoms": {
                symbol: atom.to_dict()
                for symbol, atom in self.atoms.items()
            },
            "predicates": {
                pred: {"arity": arity}
                for pred, arity in self.predicates.items()
            },
            "objects": self.objects
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundingMap':
        """Create GroundingMap from dictionary"""
        gmap = cls()

        # Load atoms
        for symbol, atom_data in data.get("atoms", {}).items():
            gmap.add_atom(
                symbol=symbol,
                predicate=atom_data["predicate"],
                args=atom_data["args"]
            )

        # predicates and objects are automatically updated by add_atom
        return gmap

    @classmethod
    def from_json(cls, json_str: str) -> 'GroundingMap':
        """Create GroundingMap from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_propositional_symbol(predicate: str, args: List[str], normalizer: SymbolNormalizer = None) -> str:
    """
    Create propositional symbol from predicate and arguments

    Uses SymbolNormalizer to handle special characters (e.g., hyphens).

    Naming convention: predicate_arg1_arg2_...
    Examples:
        on(a, b) -> on_a_b
        clear(c) -> clear_c
        handempty -> handempty
        on(block-1, block-2) -> on_blockhh1_blockhh2 (with hyphen encoding)

    Args:
        predicate: Predicate name
        args: List of arguments (may contain hyphens or other special chars)
        normalizer: Optional SymbolNormalizer instance (creates new one if not provided)

    Returns:
        Propositional symbol string with normalized encoding
    """
    if normalizer is None:
        normalizer = SymbolNormalizer()

    # Ensure all args are strings (convert if needed)
    str_args = []
    for arg in args:
        if isinstance(arg, str):
            str_args.append(arg)
        elif isinstance(arg, dict):
            # Skip nested formulas - these shouldn't be in propositional symbols
            continue
        else:
            str_args.append(str(arg))

    # Use normalizer to create symbol (handles hyphens automatically)
    return normalizer.create_propositional_symbol(predicate, str_args)


def test_grounding_map():
    """Test the GroundingMap class"""
    print("="*80)
    print("GROUNDING MAP TEST")
    print("="*80)

    # Create grounding map
    gmap = GroundingMap()

    # Add atoms
    gmap.add_atom("on_a_b", "on", ["a", "b"])
    gmap.add_atom("on_b_c", "on", ["b", "c"])
    gmap.add_atom("clear_a", "clear", ["a"])
    gmap.add_atom("handempty", "handempty", [])

    print(f"\nAtoms: {len(gmap.atoms)}")
    for symbol, atom in gmap.atoms.items():
        print(f"  {symbol} -> {atom.to_readable_format()} | PDDL: {atom.to_pddl_format()}")

    print(f"\nPredicates: {gmap.predicates}")
    print(f"Objects: {gmap.objects}")

    # Test JSON serialization
    print("\nJSON representation:")
    print(gmap.to_json())

    # Test validation
    domain_preds = ["on", "clear", "handempty", "ontable"]
    domain_objs = ["a", "b", "c"]

    print("\nValidation:")
    for symbol in gmap.atoms.keys():
        errors = gmap.validate_atom(symbol, domain_preds, domain_objs)
        if errors:
            print(f"  {symbol}: ✗ {errors}")
        else:
            print(f"  {symbol}: ✓ Valid")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_grounding_map()
