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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import get_config
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator


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

        # Results to be filled
        self.actual_ltlf: List[str] = []
        self.actual_objects: List[str] = []
        self.success: bool = False
        self.match_type: str = ""  # "exact", "partial", "wrong", "error"
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
            'actual_ltlf': ', '.join(self.actual_ltlf),
            'actual_objects': str(self.actual_objects),
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


def compare_formulas(expected: str, actual: List[str]) -> tuple[bool, str]:
    """
    Compare expected and actual formulas

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


def generate_reflection(test_result: TestResult) -> str:
    """Generate reflection on why the test failed and how to improve"""

    if test_result.match_type == "exact":
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

        # Compare with expected
        is_match, match_type = compare_formulas(
            test_case['expected_ltlf'],
            result.actual_ltlf
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
        'actual_ltlf', 'actual_objects',
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
    csv_input = script_dir / "test_cases_nl_to_ltlf_comprehensive.csv"

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output = script_dir / f"test_results_nl_to_ltlf_comprehensive_{timestamp}.csv"

    # Run tests
    run_all_tests(csv_input, csv_output)
