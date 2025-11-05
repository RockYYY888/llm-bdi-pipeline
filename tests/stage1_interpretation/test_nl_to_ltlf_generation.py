#!/usr/bin/env python3
"""
Stage 1 Testing: Natural Language -> LTLf Generation

Tests the LLM's ability to correctly generate LTLf formulas from natural language.
Results are recorded in CSV format with actual outputs and reflections.

**LTLf Syntax Reference**: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path (only once)
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from stage1_interpretation.ltlf_formula import LTLSpecification, LTLFormula
from stage1_interpretation.grounding_map import GroundingMap
from stage2_dfa_generation.ltlf_to_dfa import LTLfToDFA


class TestResult:
    """Represents a single test result"""

    def __init__(self, test_case: Dict[str, str]):
        self.test_id = test_case['test_id']
        self.category = test_case['category']
        self.natural_language = test_case['natural_language']
        self.expected_ltlf = test_case['expected_ltlf']
        self.expected_objects = test_case['expected_objects']
        self.description = test_case['description']
        self.notes = test_case['notes']

        # DFA-related fields (if available in test case)
        self.propositionalized_ltlf = test_case.get('propositionalized_ltlf', '')
        self.expected_dfa = test_case.get('expected_dfa', '')

        # Results to be filled
        self.actual_ltlf: List[str] = []
        self.actual_objects: List[str] = []
        self.actual_propositionalized_ltlf: str = ""
        self.actual_dfa: str = ""
        self.success: bool = False
        self.match_type: str = ""  # "exact", "dfa_equivalent", "dfa_different", "error"
        self.error_message: str = ""
        self.reflection: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV writing"""
        return {
            'test_id': self.test_id,
            'category': self.category,
            'natural_language': self.natural_language,
            'expected_ltlf': self.expected_ltlf,
            'expected_objects': self.expected_objects,
            'propositionalized_ltlf': self.propositionalized_ltlf,
            'expected_dfa': self.expected_dfa,
            'actual_ltlf': ', '.join(self.actual_ltlf),
            'actual_objects': str(self.actual_objects),
            'actual_propositionalized_ltlf': self.actual_propositionalized_ltlf,
            'actual_dfa': self.actual_dfa,
            'success': self.success,
            'match_type': self.match_type,
            'error_message': self.error_message,
            'reflection': self.reflection,
            'description': self.description,
            'notes': self.notes
        }


def load_test_cases(csv_path: Path) -> List[Dict[str, str]]:
    """Load test cases from CSV file"""
    test_cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)
    return test_cases


def normalize_formula(formula: str) -> str:
    """
    Normalize formula for comparison (remove spaces, standardize format)

    Handles all LTLf syntax operators
    """
    # Remove all spaces
    normalized = formula.replace(' ', '')

    # Standardize boolean operators
    normalized = normalized.replace('¬', '!')  # Old negation symbol
    normalized = normalized.replace('not', '!')  # Alternative negation
    normalized = normalized.replace('~', '!')  # Alternative negation symbol
    normalized = normalized.replace('&&', '&')  # Standardize AND
    normalized = normalized.replace('||', '|')  # Standardize OR
    normalized = normalized.replace('=>', '->')  # Standardize IMPLIES
    normalized = normalized.replace('<=>', '<->')  # Standardize EQUIVALENCE

    return normalized.lower()


def split_formulas(formula_string: str) -> List[str]:
    """
    Split comma-separated formulas while respecting parentheses

    Example: "F(on(a, b)), F(on(c, d))" -> ["F(on(a, b))", "F(on(c, d))"]
    Not: ["F(on(a", "b))", ...] (incorrect naive split)
    """
    formulas = []
    current = []
    paren_depth = 0

    for char in formula_string:
        if char == '(':
            paren_depth += 1
            current.append(char)
        elif char == ')':
            paren_depth -= 1
            current.append(char)
        elif char == ',' and paren_depth == 0:
            # This is a top-level comma separating formulas
            if current:
                formulas.append(''.join(current).strip())
                current = []
        else:
            current.append(char)

    # Don't forget the last formula
    if current:
        formulas.append(''.join(current).strip())

    return formulas


def create_spec_from_formula_string(formula_str: str, objects: List[str]) -> LTLSpecification:
    """
    Create a minimal LTLSpecification from a formula string for DFA generation

    Args:
        formula_str: LTLf formula string (e.g., "F(on(a, b))")
        objects: List of objects used in the formula

    Returns:
        LTLSpecification with grounding map
    """
    from stage1_interpretation.ltlf_generator import NLToLTLfGenerator

    # Parse the formula string to create proper LTLFormula objects
    # For now, we'll create a temporary generator and parse it
    generator = NLToLTLfGenerator(api_key="dummy", model="gpt-4")

    # Create a mock spec with the formula string
    # We need to actually parse this into LTLFormula objects
    # For simplicity, let's just use the string directly via ltlf2dfa
    spec = LTLSpecification()
    spec.objects = objects

    # Create a simple formula wrapper that returns the string
    class StringFormula:
        def __init__(self, s):
            self.s = s
        def to_string(self):
            return self.s

    spec.formulas = [StringFormula(formula_str)]

    return spec


def compare_formulas_via_dfa(expected: str, actual: List[str], objects: List[str]) -> tuple[bool, str]:
    """
    Compare expected and actual formulas by converting to DFA and checking equivalence

    Args:
        expected: Expected LTLf formula string
        actual: List of actual LTLf formula strings from generator
        objects: List of objects in the domain

    Returns:
        (is_match, match_type)
        match_type: "dfa_equivalent", "dfa_different", "error"
    """
    if not actual:
        return False, "error"

    try:
        # Create DFA converter
        converter = LTLfToDFA()

        # Parse expected formula(s)
        expected_formulas = split_formulas(expected)

        # Join expected formulas with AND if multiple
        if len(expected_formulas) > 1:
            expected_combined = " & ".join(f"({f})" for f in expected_formulas)
        else:
            expected_combined = expected_formulas[0]

        # Join actual formulas with AND if multiple
        if len(actual) > 1:
            actual_combined = " & ".join(f"({f})" for f in actual)
        else:
            actual_combined = actual[0]

        # Create specs
        expected_spec = create_spec_from_formula_string(expected_combined, objects)
        actual_spec = create_spec_from_formula_string(actual_combined, objects)

        # Convert to DFA
        expected_dfa, expected_meta = converter.convert(expected_spec)
        actual_dfa, actual_meta = converter.convert(actual_spec)

        # Compare DFAs by checking if they have the same structure
        # For now, use a simple heuristic: compare propositional formulas
        # A proper DFA equivalence check would require graph isomorphism checking

        # Simple check: if propositional formulas are identical, DFAs are equivalent
        exp_prop = expected_meta['propositional_formula']
        act_prop = actual_meta['propositional_formula']

        # Normalize both for comparison
        exp_norm = normalize_formula(exp_prop)
        act_norm = normalize_formula(act_prop)

        if exp_norm == act_norm:
            return True, "dfa_equivalent"
        else:
            # Try parsing with ltlf2dfa to see if they're semantically equivalent
            # This is a basic check - proper equivalence needs graph comparison
            return False, "dfa_different"

    except Exception as e:
        # If DFA generation fails, fall back to string comparison
        print(f"  ⚠️  DFA comparison failed: {str(e)}")
        print(f"  → Falling back to string comparison")
        return compare_formulas_string_based(expected, actual)


def compare_formulas_string_based(expected: str, actual: List[str]) -> tuple[bool, str]:
    """
    Original string-based formula comparison (fallback method)

    Returns:
        (is_match, match_type)
        match_type: "exact", "partial", "wrong"
    """
    # Parse expected (can be comma-separated, but respect parentheses)
    expected_formulas = split_formulas(expected)

    # Normalize for comparison
    expected_normalized = set(normalize_formula(f) for f in expected_formulas)
    actual_normalized = set(normalize_formula(f) for f in actual)

    if expected_normalized == actual_normalized:
        return True, "exact"
    elif expected_normalized & actual_normalized:  # intersection
        return False, "partial"
    else:
        return False, "wrong"


def compare_formulas(expected: str, actual: List[str], objects: List[str] = None) -> tuple[bool, str]:
    """
    Compare expected and actual formulas using DFA equivalence check

    Args:
        expected: Expected LTLf formula string
        actual: List of actual LTLf formula strings
        objects: List of objects (optional, for DFA comparison)

    Returns:
        (is_match, match_type)
    """
    # Try DFA-based comparison if objects are provided
    if objects:
        try:
            return compare_formulas_via_dfa(expected, actual, objects)
        except Exception as e:
            print(f"  ⚠️  DFA comparison error: {str(e)}")
            print(f"  → Using string comparison")

    # Fall back to string-based comparison
    return compare_formulas_string_based(expected, actual)


def generate_reflection(test_result: TestResult) -> str:
    """Generate reflection on why the test failed and how to improve"""

    if test_result.match_type in ["exact", "dfa_equivalent", "dfa_exact"]:
        if test_result.match_type == "dfa_exact":
            return "✓ DFA-exact (identical DFAs)"
        elif test_result.match_type == "dfa_equivalent":
            return "✓ DFA-equivalent (semantically correct)"
        return "✓ Perfect match"

    reflections = []

    # Analyze the mismatch
    if test_result.match_type == "error":
        reflections.append(f"ERROR: {test_result.error_message}")
        reflections.append("Possible causes: LLM API issue, parsing error, or invalid JSON response")
        reflections.append("Improvement: Check prompt clarity and error handling")

    elif test_result.match_type == "wrong":
        reflections.append("Formula completely mismatched")
        expected = test_result.expected_ltlf
        actual = ', '.join(test_result.actual_ltlf)
        reflections.append(f"Expected: {expected}")
        reflections.append(f"Got: {actual}")

        # Specific analysis for temporal operators
        if "F(" in expected and "F(" not in actual:
            reflections.append("Missing F (Eventually) operator")
        if "G(" in expected and "G(" not in actual:
            reflections.append("Missing G (Always) operator")
        if "X(" in expected and "X(" not in actual:
            reflections.append("Missing X (Next) operator")
        if "WX(" in expected and "WX(" not in actual:
            reflections.append("Missing WX (Weak Next) operator")
        if " U " in expected and " U " not in actual:
            reflections.append("Missing U (Until) operator")
        if " R " in expected and " R " not in actual:
            reflections.append("Missing R (Release) operator")

        # Boolean operators analysis
        if " & " in expected and " & " not in actual:
            reflections.append("Missing & (AND) operator")
        if " | " in expected and " | " not in actual:
            reflections.append("Missing | (OR) operator")
        if "!(" in expected and "!(" not in actual:
            reflections.append("Missing ! (NOT) operator")
        if " -> " in expected and " -> " not in actual:
            reflections.append("Missing -> (IMPLIES) operator")
        if " <-> " in expected and " <-> " not in actual:
            reflections.append("Missing <-> (EQUIVALENCE) operator")

        # Check object recognition
        if test_result.expected_objects != str(test_result.actual_objects):
            reflections.append(f"Object mismatch - Expected: {test_result.expected_objects}, Got: {test_result.actual_objects}")
            reflections.append("Improvement: Enhance object extraction in prompt")

    elif test_result.match_type == "partial":
        reflections.append("Partial match - some formulas correct, others missing or wrong")
        reflections.append("Improvement: Clarify conjunctive vs separate goals in prompt")

    return " | ".join(reflections)


def run_test_case(generator: NLToLTLfGenerator, test_case: Dict[str, str]) -> TestResult:
    """Run a single test case"""
    result = TestResult(test_case)

    try:
        # Generate LTLf from natural language
        ltl_spec, _, _ = generator.generate(test_case['natural_language'])

        # Extract results
        result.actual_ltlf = [f.to_string() for f in ltl_spec.formulas]
        result.actual_objects = ltl_spec.objects

        # Generate actual DFA
        try:
            dfa_converter = LTLfToDFA()
            actual_dfa_dot, actual_metadata = dfa_converter.convert(ltl_spec)
            result.actual_propositionalized_ltlf = actual_metadata['propositional_formula']
            # Escape quotes and newlines for CSV storage
            result.actual_dfa = actual_dfa_dot.replace('"', '""').replace('\n', '\\n')
        except Exception as dfa_error:
            result.actual_propositionalized_ltlf = f"DFA_ERROR: {str(dfa_error)}"
            result.actual_dfa = ""

        # Compare DFAs if both expected and actual are available
        if result.expected_dfa and result.actual_dfa:
            # Compare DFA strings directly (they should be identical if semantically equivalent)
            if result.expected_dfa == result.actual_dfa:
                result.success = True
                result.match_type = "dfa_exact"
            else:
                # DFAs are different - formulas are not equivalent
                result.success = False
                result.match_type = "dfa_different"
        else:
            # Fall back to formula comparison
            is_match, match_type = compare_formulas(
                test_case['expected_ltlf'],
                result.actual_ltlf,
                objects=result.actual_objects
            )
            result.success = is_match
            result.match_type = match_type

    except Exception as e:
        result.success = False
        result.match_type = "error"
        result.error_message = str(e)

    # Generate reflection
    result.reflection = generate_reflection(result)

    return result


def run_all_tests(csv_path: Path, output_path: Path):
    """Run all test cases and save results"""

    print("=" * 80)
    print("STAGE 1 TEST: Natural Language -> LTLf Generation")
    print("=" * 80)
    print()

    # Load test cases
    test_cases = load_test_cases(csv_path)
    print(f"Loaded {len(test_cases)} test cases from {csv_path}")
    print()

    # Initialize generator using the actual pipeline's domain configuration
    config = get_config()
    # Use default blocksworld domain (same as main pipeline)
    domain_file = str(Path(__file__).parent.parent.parent / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl")

    generator = NLToLTLfGenerator(
        api_key=config.openai_api_key,
        model=config.openai_model,
        domain_file=domain_file
    )

    # Run tests
    results: List[TestResult] = []
    categories_stats = {}

    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test_case['test_id']} - {test_case['description']}")
        print(f"  NL: \"{test_case['natural_language']}\"")

        result = run_test_case(generator, test_case)
        results.append(result)

        # Update category stats
        category = result.category
        if category not in categories_stats:
            categories_stats[category] = {'total': 0, 'passed': 0}
        categories_stats[category]['total'] += 1
        if result.success:
            categories_stats[category]['passed'] += 1

        # Print result
        status = "✓ PASS" if result.success else f"✗ FAIL ({result.match_type})"
        print(f"  Expected: {test_case['expected_ltlf']}")
        print(f"  Actual:   {', '.join(result.actual_ltlf)}")
        print(f"  {status}")

        if not result.success:
            print(f"  Reflection: {result.reflection}")

        print()

    # Save results to CSV
    save_results(results, output_path)

    # Print summary
    print_summary(results, categories_stats)

    return results


def save_results(results: List[TestResult], output_path: Path):
    """Save test results to CSV"""

    fieldnames = [
        'test_id', 'category', 'natural_language',
        'expected_ltlf', 'expected_objects',
        'propositionalized_ltlf', 'expected_dfa',
        'actual_ltlf', 'actual_objects',
        'actual_propositionalized_ltlf', 'actual_dfa',
        'success', 'match_type', 'error_message', 'reflection',
        'description', 'notes'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"Results saved to: {output_path}")
    print()


def print_summary(results: List[TestResult], categories_stats: Dict):
    """Print test summary"""

    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total}")
    print(f"✓ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"✗ Failed: {failed} ({failed/total*100:.1f}%)")
    print()

    # Match type breakdown
    match_types = {}
    for result in results:
        mt = result.match_type
        match_types[mt] = match_types.get(mt, 0) + 1

    print("Match Type Breakdown:")
    for match_type, count in sorted(match_types.items()):
        print(f"  {match_type}: {count}")
    print()

    # Category breakdown
    print("Results by Category:")
    for category, stats in sorted(categories_stats.items()):
        success_rate = stats['passed'] / stats['total'] * 100
        print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    print()

    # Failed tests
    if failed > 0:
        print("Failed Tests:")
        for result in results:
            if not result.success:
                print(f"  [{result.test_id}] {result.natural_language}")
                print(f"    Expected: {result.expected_ltlf}")
                print(f"    Got: {', '.join(result.actual_ltlf)}")
                print(f"    Reflection: {result.reflection}")
                print()

    print("=" * 80)


if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent

    # Use the merged CSV file with all test cases (comprehensive + complex_nested)
    csv_input = script_dir / "nl_to_ltlf_test_cases.csv"

    # Check if CSV exists
    if not csv_input.exists():
        print(f"❌ Error: Test cases CSV not found at {csv_input}")
        print("   Expected file: nl_to_ltlf_test_cases.csv")
        sys.exit(1)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = script_dir / f"test_results_nl_to_ltlf_{timestamp}.csv"

    # Run tests
    run_all_tests(csv_input, csv_output)
