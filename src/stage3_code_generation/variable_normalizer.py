"""
Variable Normalizer for Backward Planning (Schema-Level Abstraction)

This module provides functionality to normalize grounded predicates into
schema-level variable predicates for backward planning optimization.

Key idea (Position-Based Normalization):
- on(a, b) → on(?arg0, ?arg1)
- on(c, d) → on(?arg0, ?arg1)  ✓ SAME SCHEMA!
- on(b, a) → on(?arg0, ?arg1)  ✓ SAME SCHEMA!

All goals with the same predicate structure share the same abstract plan,
regardless of which specific objects are involved. This achieves TRUE
schema-level abstraction and massive cache hit rates.
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from stage3_code_generation.state_space import PredicateAtom
from utils.pddl_parser import PDDLDomain


@dataclass
class VariableMapping:
    """
    Represents a mapping between objects and variables

    Attributes:
        obj_to_var: Object name → Variable name (e.g., {"a": "?v0", "b": "?v1"})
        var_to_obj: Variable name → Object name (e.g., {"?v0": "a", "?v1": "b"})
    """
    obj_to_var: Dict[str, str]
    var_to_obj: Dict[str, str]


class VariableNormalizer:
    """
    Normalizes grounded predicates to schema-level variable predicates

    Uses POSITION-BASED normalization for TRUE schema-level abstraction:
    - on(a, b) → on(?arg0, ?arg1)
    - on(c, d) → on(?arg0, ?arg1)  ✓ SAME SCHEMA - Cache hit!
    - on(b, a) → on(?arg0, ?arg1)  ✓ SAME SCHEMA - Cache hit!

    All goals with same predicate structure share the same abstract plan,
    regardless of which specific objects are used.

    Key Insight: We care about the STRUCTURE (on(?X, ?Y)), not the
    specific objects (a, b vs c, d). This is the essence of lifted planning.

    Attributes:
        domain: PDDL domain (for type information)
        object_list: List of all objects in the problem (for reference)
        object_types: Map from object name to type
    """

    def __init__(self, domain: PDDLDomain, objects: List[str]):
        """
        Initialize variable normalizer

        Args:
            domain: PDDL domain
            objects: List of objects in the problem
        """
        self.domain = domain
        self.object_list = sorted(objects)  # Sort for consistency

        # Extract types from domain
        # For blocksworld, all objects are of type "block"
        # In general, we'd need to infer or receive type information
        self.object_types = self._infer_object_types()

    def _infer_object_types(self) -> Dict[str, str]:
        """
        Infer object types from domain

        For blocksworld, all objects are "block" type.
        In future, this could be extended to handle multiple types.

        Returns:
            Map from object name to type name
        """
        # Simple implementation: assign all objects to first domain type
        if self.domain.types:
            default_type = self.domain.types[0]
        else:
            default_type = "object"

        return {obj: default_type for obj in self.object_list}

    def _is_constant(self, arg: str) -> bool:
        """
        Determine if an argument is a constant (not an object to abstract)

        Constants include:
        - Already variables: ?var, ?x
        - Numbers: -2, 3.14, 10
        - String literals with quotes: 'Left', "Right"
        - Boolean/null: true, false, nil, null
        - Uppercase identifiers (often constants): LEFT, RIGHT, UP, DOWN

        Objects (to abstract):
        - Items in object_list
        - Lowercase identifiers not matching above patterns

        Args:
            arg: Argument string to check

        Returns:
            True if argument is a constant (should NOT be abstracted)
        """
        # Already a variable - keep as-is
        if arg.startswith('?'):
            return True

        # Try parsing as number (int or float)
        try:
            float(arg)
            return True
        except ValueError:
            pass

        # String literals with quotes
        if (arg.startswith("'") and arg.endswith("'")) or \
           (arg.startswith('"') and arg.endswith('"')):
            return True

        # Boolean/null keywords
        if arg.lower() in ['true', 'false', 'nil', 'null', 'none']:
            return True

        # Uppercase identifiers (common constant convention)
        # e.g., LEFT, RIGHT, UP, DOWN
        if arg.isupper() and arg.isalpha():
            return True

        # Otherwise, treat as object to be abstracted
        return False

    def normalize_predicates(self, predicates: List[PredicateAtom]) -> Tuple[List[PredicateAtom], VariableMapping]:
        """
        Normalize a list of grounded predicates to variable form using SCHEMA-LEVEL abstraction

        Strategy (Position-Based Normalization with Constant Preservation):
        1. Identify constants vs objects in arguments
        2. Abstract only OBJECTS (preserve constants)
        3. Assign variables based on FIRST OCCURRENCE ORDER of objects
        4. Use ?arg0, ?arg1, ?arg2, ... for objects only
        5. Same object in different predicates maps to SAME variable

        This enables TRUE schema-level caching where goals with different objects
        but same structure AND constants share the same abstract plan!

        Examples:
            Input: [on(a, b)]             → Output: [on(?arg0, ?arg1)]
            Input: [on(c, d)]             → Output: [on(?arg0, ?arg1)]  ✓ SAME SCHEMA!
            Input: [move(a, -2, 'Left')]  → Output: [move(?arg0, -2, 'Left')]  ✓ Constants preserved!
            Input: [move(b, -2, 'Left')]  → Output: [move(?arg0, -2, 'Left')]  ✓ Shares schema!
            Input: [move(a, 5, 'Right')]  → Output: [move(?arg0, 5, 'Right')]  ✓ Different schema (diff constants)!

        Key Insight: We abstract object IDENTITIES, not constant VALUES.
        Constants are part of the schema definition.

        Args:
            predicates: List of grounded predicates

        Returns:
            Tuple of (normalized predicates, variable mapping)
        """
        # SCHEMA-LEVEL: Assign variables based on first occurrence order
        # BUT only for objects, not constants
        obj_to_var = {}
        var_counter = 0

        # First pass: collect OBJECTS (not constants) in order of first appearance
        for pred in predicates:
            for arg in pred.args:
                # Skip constants and already-mapped objects
                if self._is_constant(arg) or arg in obj_to_var:
                    continue

                # This is an object - assign a variable
                obj_to_var[arg] = f"?arg{var_counter}"
                var_counter += 1

        # Create reverse mapping (only for objects that were abstracted)
        var_to_obj = {var: obj for obj, var in obj_to_var.items()}

        mapping = VariableMapping(obj_to_var=obj_to_var, var_to_obj=var_to_obj)

        # Second pass: normalize predicates
        normalized_predicates = []
        for pred in predicates:
            new_args = []
            for arg in pred.args:
                if self._is_constant(arg):
                    # Keep constants as-is
                    new_args.append(arg)
                else:
                    # Replace objects with schema variables
                    new_args.append(obj_to_var.get(arg, arg))

            normalized_pred = PredicateAtom(pred.name, new_args, pred.negated)
            normalized_predicates.append(normalized_pred)

        return normalized_predicates, mapping

    def denormalize_predicates(self, predicates: List[PredicateAtom],
                               mapping: VariableMapping) -> List[PredicateAtom]:
        """
        Convert variable-based predicates back to grounded form

        Example:
            Input: [on(?v0, ?v1), clear(?v0)] + mapping {?v0: a, ?v1: b}
            Output: [on(a, b), clear(a)]

        Args:
            predicates: List of variable-based predicates
            mapping: Variable mapping

        Returns:
            List of grounded predicates
        """
        grounded_predicates = []
        for pred in predicates:
            # Replace variables with objects
            new_args = [mapping.var_to_obj.get(arg, arg) for arg in pred.args]
            grounded_pred = PredicateAtom(pred.name, new_args, pred.negated)
            grounded_predicates.append(grounded_pred)

        return grounded_predicates

    def serialize_goal(self, predicates: List[PredicateAtom]) -> str:
        """
        Serialize predicates to a canonical string form for caching

        This is used as the cache key for variable-level goal caching.

        Example:
            Input: [on(?v0, ?v1), clear(?v2)]
            Output: "clear(?v2)|on(?v0, ?v1)"  (sorted)

        Args:
            predicates: List of predicates (should be normalized)

        Returns:
            Canonical string representation
        """
        # Sort predicates for canonical form
        sorted_preds = sorted([p.to_agentspeak() for p in predicates])
        return "|".join(sorted_preds)

    def get_variable_list(self, num_objects: int) -> List[str]:
        """
        Get list of variables for a given number of objects

        Args:
            num_objects: Number of objects/variables needed

        Returns:
            List of variable names: ["?v0", "?v1", "?v2", ...]
        """
        return [f"?v{i}" for i in range(num_objects)]

    def create_mapping_from_lists(self, objects: List[str], variables: List[str]) -> VariableMapping:
        """
        Create variable mapping from parallel lists

        Args:
            objects: List of object names
            variables: List of variable names (same length as objects)

        Returns:
            VariableMapping
        """
        assert len(objects) == len(variables), "Objects and variables must have same length"

        obj_to_var = {obj: var for obj, var in zip(objects, variables)}
        var_to_obj = {var: obj for obj, var in zip(objects, variables)}

        return VariableMapping(obj_to_var=obj_to_var, var_to_obj=var_to_obj)


def test_variable_normalizer():
    """Test variable normalizer"""
    print("="*80)
    print("Testing Variable Normalizer")
    print("="*80)

    from utils.pddl_parser import PDDLParser

    # Load domain
    domain_file = Path(__file__).parent.parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        print("Skipping test")
        return

    domain = PDDLParser.parse_domain(str(domain_file))
    print(f"Loaded domain: {domain.name}")
    print(f"Types: {domain.types}")

    # Create normalizer
    objects = ["a", "b", "c"]
    normalizer = VariableNormalizer(domain, objects)

    print(f"\nObjects: {objects}")
    print(f"Object types: {normalizer.object_types}")

    # Test 1: Normalize goal predicates
    print("\n" + "="*80)
    print("Test 1: Normalize goal predicates")
    print("="*80)

    goal1 = [PredicateAtom("on", ["a", "b"])]
    print(f"\nGoal 1 (original): {[str(p) for p in goal1]}")

    normalized1, mapping1 = normalizer.normalize_predicates(goal1)
    print(f"Goal 1 (normalized): {[str(p) for p in normalized1]}")
    print(f"Mapping: {mapping1.obj_to_var}")
    print(f"Serialized: {normalizer.serialize_goal(normalized1)}")

    goal2 = [PredicateAtom("on", ["c", "d"])]
    print(f"\nGoal 2 (original): {[str(p) for p in goal2]}")

    # Note: d is not in objects list, but normalizer should handle it
    normalized2, mapping2 = normalizer.normalize_predicates(goal2)
    print(f"Goal 2 (normalized): {[str(p) for p in normalized2]}")
    print(f"Mapping: {mapping2.obj_to_var}")
    print(f"Serialized: {normalizer.serialize_goal(normalized2)}")

    # Check if they produce the same serialized form
    key1 = normalizer.serialize_goal(normalized1)
    key2 = normalizer.serialize_goal(normalized2)
    print(f"\nSame cache key? {key1 == key2}")

    # Test 2: Complex goal with multiple predicates
    print("\n" + "="*80)
    print("Test 2: Complex goal")
    print("="*80)

    goal3 = [
        PredicateAtom("on", ["a", "b"]),
        PredicateAtom("clear", ["a"]),
        PredicateAtom("ontable", ["c"])
    ]
    print(f"\nGoal 3 (original): {[str(p) for p in goal3]}")

    normalized3, mapping3 = normalizer.normalize_predicates(goal3)
    print(f"Goal 3 (normalized): {[str(p) for p in normalized3]}")
    print(f"Mapping: {mapping3.obj_to_var}")
    print(f"Serialized: {normalizer.serialize_goal(normalized3)}")

    # Verify consistency: 'a' should map to same variable in both predicates
    a_var_in_on = None
    a_var_in_clear = None
    for pred in normalized3:
        if pred.name == "on" and len(pred.args) >= 1:
            a_var_in_on = pred.args[0]
        if pred.name == "clear" and len(pred.args) >= 1:
            a_var_in_clear = pred.args[0]

    print(f"\nConsistency check:")
    print(f"  'a' in on(...): {a_var_in_on}")
    print(f"  'a' in clear(...): {a_var_in_clear}")
    print(f"  Consistent? {a_var_in_on == a_var_in_clear}")

    # Test 3: Denormalization
    print("\n" + "="*80)
    print("Test 3: Denormalization")
    print("="*80)

    denormalized3 = normalizer.denormalize_predicates(normalized3, mapping3)
    print(f"\nOriginal: {[str(p) for p in goal3]}")
    print(f"Normalized: {[str(p) for p in normalized3]}")
    print(f"Denormalized: {[str(p) for p in denormalized3]}")

    # Check if denormalization recovers original
    original_set = set(goal3)
    denormalized_set = set(denormalized3)
    print(f"Denormalization correct? {original_set == denormalized_set}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_variable_normalizer()
