"""
Variable Normalizer for Backward Planning

This module provides functionality to normalize grounded predicates into
variable-based predicates for backward planning optimization.

Key idea: on(a, b) and on(c, d) should both normalize to on(?v0, ?v1),
allowing them to share the same state space exploration.
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
    Normalizes grounded predicates to variable-based predicates

    This enables variable-level caching in backward planning:
    - on(a, b) → on(?v0, ?v1)
    - on(c, d) → on(?v0, ?v1)

    Both map to the same variable pattern and can share state graphs.

    Attributes:
        domain: PDDL domain (for type information)
        object_list: List of all objects in the problem
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

    def normalize_predicates(self, predicates: List[PredicateAtom]) -> Tuple[List[PredicateAtom], VariableMapping]:
        """
        Normalize a list of grounded predicates to variable form

        Strategy:
        1. Use GLOBAL object list (from self.object_list) for consistent variable assignment
        2. Assign variables based on global object order: ?v0, ?v1, ?v2, ...
        3. Replace object arguments with corresponding variables

        This ensures that the same objects ALWAYS map to the same variables,
        regardless of the order they appear in the predicates.

        Example (with global objects=['a', 'b', 'c']):
            Input: [on(a, b)]     → Output: [on(?v0, ?v1)]
            Input: [on(b, a)]     → Output: [on(?v1, ?v0)]
            Input: [on(c, b)]     → Output: [on(?v2, ?v1)]

        Args:
            predicates: List of grounded predicates

        Returns:
            Tuple of (normalized predicates, variable mapping)
        """
        # Use GLOBAL object list for consistent variable assignment
        # This ensures a→?v0, b→?v1, c→?v2 etc. regardless of predicate order
        obj_to_var = {obj: f"?v{i}" for i, obj in enumerate(self.object_list)}
        var_to_obj = {var: obj for obj, var in obj_to_var.items()}

        mapping = VariableMapping(obj_to_var=obj_to_var, var_to_obj=var_to_obj)

        # Normalize each predicate
        normalized_predicates = []
        for pred in predicates:
            # Replace object arguments with variables
            new_args = [obj_to_var.get(arg, arg) for arg in pred.args]
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
