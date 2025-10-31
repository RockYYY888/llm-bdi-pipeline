"""
PDDL Domain Parser

Extracts domain information from PDDL domain files:
- Domain name
- Types
- Predicates with their parameters
- Actions with parameters

This information is used to dynamically construct LLM prompts.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class PDDLPredicate:
    """Represents a PDDL predicate"""
    name: str
    parameters: List[str]  # e.g., ["?b - block"]

    def to_signature(self) -> str:
        """Convert to signature format: on(X, Y)"""
        if not self.parameters:
            return self.name

        # Extract variable names (without types)
        var_names = []
        for param in self.parameters:
            # Format: "?b - block" or "?b1 ?b2 - block"
            parts = param.split('-')
            if len(parts) >= 1:
                vars_part = parts[0].strip()
                # Extract variable names (remove ?)
                vars_in_this_part = [v.strip().lstrip('?').upper()
                                     for v in vars_part.split() if v.strip()]
                var_names.extend(vars_in_this_part)

        return f"{self.name}({', '.join(var_names)})"


@dataclass
class PDDLAction:
    """Represents a PDDL action"""
    name: str
    parameters: List[str]


@dataclass
class PDDLDomain:
    """Parsed PDDL domain information"""
    name: str
    types: List[str]
    predicates: List[PDDLPredicate]
    actions: List[PDDLAction]

    def get_predicate_signatures(self) -> List[str]:
        """Get all predicate signatures in format: on(X, Y)"""
        return [p.to_signature() for p in self.predicates]

    def get_action_names(self) -> List[str]:
        """Get all action names"""
        return [a.name for a in self.actions]


class PDDLParser:
    """Parser for PDDL domain files"""

    @staticmethod
    def parse_domain(file_path: str) -> PDDLDomain:
        """
        Parse a PDDL domain file

        Args:
            file_path: Path to domain.pddl file

        Returns:
            PDDLDomain with parsed information
        """
        with open(file_path, 'r') as f:
            content = f.read()

        # Remove comments
        content = re.sub(r';.*$', '', content, flags=re.MULTILINE)

        # Extract domain name
        domain_name_match = re.search(r'\(define\s+\(domain\s+(\w+)\)', content)
        domain_name = domain_name_match.group(1) if domain_name_match else "unknown"

        # Extract types
        types = PDDLParser._extract_types(content)

        # Extract predicates
        predicates = PDDLParser._extract_predicates(content)

        # Extract actions
        actions = PDDLParser._extract_actions(content)

        return PDDLDomain(
            name=domain_name,
            types=types,
            predicates=predicates,
            actions=actions
        )

    @staticmethod
    def _extract_types(content: str) -> List[str]:
        """Extract type declarations"""
        types_match = re.search(r'\(:types\s+([^)]+)\)', content)
        if not types_match:
            return []

        types_str = types_match.group(1)
        # Simple extraction (e.g., "block" from ":types block")
        types = [t.strip() for t in types_str.split() if t.strip()]
        return types

    @staticmethod
    def _extract_predicates(content: str) -> List[PDDLPredicate]:
        """Extract predicate declarations"""
        predicates_match = re.search(r'\(:predicates\s+(.*?)\s*\)\s*(?:\(:action|\(:derived|$)',
                                      content, re.DOTALL)
        if not predicates_match:
            return []

        predicates_str = predicates_match.group(1)

        # Find all predicate definitions: (predicate_name ?args)
        predicate_pattern = r'\(([a-zA-Z][a-zA-Z0-9_-]*)\s*([^)]*)\)'
        matches = re.findall(predicate_pattern, predicates_str)

        predicates = []
        for pred_name, params_str in matches:
            params = PDDLParser._parse_parameters(params_str)
            predicates.append(PDDLPredicate(name=pred_name, parameters=params))

        return predicates

    @staticmethod
    def _extract_actions(content: str) -> List[PDDLAction]:
        """Extract action declarations"""
        # Find all action definitions
        action_pattern = r'\(:action\s+([a-zA-Z][a-zA-Z0-9_-]*)\s+:parameters\s+\(([^)]*)\)'
        matches = re.findall(action_pattern, content, re.DOTALL)

        actions = []
        for action_name, params_str in matches:
            params = PDDLParser._parse_parameters(params_str)
            actions.append(PDDLAction(name=action_name, parameters=params))

        return actions

    @staticmethod
    def _parse_parameters(params_str: str) -> List[str]:
        """
        Parse parameter string

        Examples:
        - "?b - block" → ["?b - block"]
        - "?b1 ?b2 - block" → ["?b1 - block", "?b2 - block"]
        """
        if not params_str or not params_str.strip():
            return []

        params_str = params_str.strip()

        # Handle typed parameters: ?x ?y - type
        if '-' in params_str:
            parts = params_str.split('-')
            if len(parts) >= 2:
                vars_part = parts[0].strip()
                type_part = parts[1].strip().split()[0] if parts[1].strip() else "object"

                # Extract individual variables
                variables = [v.strip() for v in vars_part.split() if v.strip()]

                # Return each variable with its type
                return [f"{var} - {type_part}" for var in variables]

        # Untyped parameters
        return [params_str]


def test_pddl_parser():
    """Test the PDDL parser"""
    import sys
    from pathlib import Path

    # Find blocksworld domain
    domain_file = Path(__file__).parent / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"

    if not domain_file.exists():
        print(f"Domain file not found: {domain_file}")
        return

    print("="*80)
    print("PDDL PARSER TEST")
    print("="*80)
    print(f"Parsing: {domain_file}")
    print()

    domain = PDDLParser.parse_domain(str(domain_file))

    print(f"Domain Name: {domain.name}")
    print(f"\nTypes: {domain.types}")
    print(f"\nPredicates ({len(domain.predicates)}):")
    for pred in domain.predicates:
        print(f"  - {pred.to_signature()}")

    print(f"\nActions ({len(domain.actions)}):")
    for action in domain.actions:
        print(f"  - {action.name}")

    print()
    print("="*80)


if __name__ == "__main__":
    test_pddl_parser()
