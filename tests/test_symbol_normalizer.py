"""
Tests for SymbolNormalizer

Verifies hyphen encoding/decoding and integration with the pipeline.
"""

import sys
from pathlib import Path

# Add src to path (only once)
_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.symbol_normalizer import SymbolNormalizer
from stage1_interpretation.grounding_map import create_propositional_symbol
from stage2_dfa_generation.ltlf_to_dfa import PredicateToProposition


def test_hyphen_encoding_decoding():
    """Test basic hyphen encoding and decoding"""
    print("=" * 80)
    print("TEST 1: Hyphen Encoding/Decoding")
    print("=" * 80)

    normalizer = SymbolNormalizer()

    test_cases = [
        ("block-1", "blockhh1"),
        ("rover-a", "roverhha"),
        ("location-base", "locationhhbase"),
        ("multi-word-name", "multihhwordhhname"),
    ]

    all_passed = True
    for original, expected_encoded in test_cases:
        encoded = normalizer.encode_hyphens(original)
        decoded = normalizer.decode_hyphens(encoded)

        if encoded == expected_encoded and decoded == original:
            print(f"✓ {original:20} → {encoded:25} → {decoded:20}")
        else:
            print(f"✗ {original:20} → {encoded:25} (expected: {expected_encoded})")
            all_passed = False

    print()
    return all_passed


def test_propositional_symbol_with_hyphens():
    """Test propositional symbol creation with hyphenated arguments"""
    print("=" * 80)
    print("TEST 2: Propositional Symbol Creation with Hyphens")
    print("=" * 80)

    normalizer = SymbolNormalizer()

    test_cases = [
        ("on", ["block-1", "block-2"], "on_blockhh1_blockhh2"),
        ("at", ["rover-a", "location-base"], "at_roverhha_locationhhbase"),
        ("clear", ["block-1"], "clear_blockhh1"),
        ("holding", ["object-x"], "holding_objecthhx"),
    ]

    all_passed = True
    for predicate, args, expected in test_cases:
        result = normalizer.create_propositional_symbol(predicate, args)

        if result == expected:
            print(f"✓ {predicate}({', '.join(args):30}) → {result:35}")
        else:
            print(f"✗ {predicate}({', '.join(args):30}) → {result:35} (expected: {expected})")
            all_passed = False

    print()
    return all_passed


def test_formula_normalization_with_hyphens():
    """Test LTLf formula normalization with hyphenated predicates"""
    print("=" * 80)
    print("TEST 3: Formula Normalization with Hyphens")
    print("=" * 80)

    normalizer = SymbolNormalizer()

    test_cases = [
        ("F(on(block-1, block-2))", "F(on_blockhh1_blockhh2)"),
        ("G(clear(block-1))", "G(clear_blockhh1)"),
        ("F(at(rover-a, location-base))", "F(at_roverhha_locationhhbase)"),
        ("F(on(block-1, table)) & G(clear(block-2))", "F(on_blockhh1_table) & G(clear_blockhh2)"),
        ("(on(block-1, table) U clear(block-2))", "(on_blockhh1_table U clear_blockhh2)"),
    ]

    all_passed = True
    for original, expected in test_cases:
        result = normalizer.normalize_formula_string(original)

        if result == expected:
            print(f"✓ {original:50}")
            print(f"  → {result:50}")
        else:
            print(f"✗ {original:50}")
            print(f"  → {result:50}")
            print(f"  Expected: {expected:50}")
            all_passed = False

    print()
    return all_passed


def test_symbol_restoration():
    """Test restoring hyphens from normalized symbols"""
    print("=" * 80)
    print("TEST 4: Symbol Restoration (Denormalization)")
    print("=" * 80)

    normalizer = SymbolNormalizer()

    # First create normalized symbols to build mapping
    test_pairs = [
        (("on", ["block-1", "block-2"]), "on_block-1_block-2"),
        (("at", ["rover-a", "location-base"]), "at_rover-a_location-base"),
        (("clear", ["block-1"]), "clear_block-1"),
    ]

    all_passed = True
    for (predicate, args), expected_restored in test_pairs:
        normalized = normalizer.create_propositional_symbol(predicate, args)
        restored = normalizer.restore_symbol_hyphens(normalized)

        if restored == expected_restored:
            print(f"✓ {normalized:35} → {restored:35}")
        else:
            print(f"✗ {normalized:35} → {restored:35} (expected: {expected_restored})")
            all_passed = False

    print()
    return all_passed


def test_integration_with_predicate_to_proposition():
    """Test integration with PredicateToProposition class"""
    print("=" * 80)
    print("TEST 5: Integration with PredicateToProposition")
    print("=" * 80)

    normalizer = SymbolNormalizer()
    converter = PredicateToProposition(normalizer)

    test_formulas = [
        ("F(on(block-1, block-2))", "F(on_blockhh1_blockhh2)"),
        ("G(clear(rover-a))", "G(clear_roverhha)"),
        ("F(on(a, b)) & G(at(rover-1, base))", "F(on_a_b) & G(at_roverhh1_base)"),
    ]

    all_passed = True
    for original, expected in test_formulas:
        result = converter.convert_formula(original)

        if result == expected:
            print(f"✓ {original:50}")
            print(f"  → {result:50}")
        else:
            print(f"✗ {original:50}")
            print(f"  → {result:50}")
            print(f"  Expected: {expected:50}")
            all_passed = False

    print()
    return all_passed


def test_grounding_map_integration():
    """Test integration with create_propositional_symbol"""
    print("=" * 80)
    print("TEST 6: Integration with Grounding Map")
    print("=" * 80)

    normalizer = SymbolNormalizer()

    test_cases = [
        ("on", ["block-1", "block-2"], "on_blockhh1_blockhh2"),
        ("at", ["rover-a", "base"], "at_roverhha_base"),
        ("clear", ["object-x"], "clear_objecthhx"),
    ]

    all_passed = True
    for predicate, args, expected in test_cases:
        result = create_propositional_symbol(predicate, args, normalizer)

        if result == expected:
            print(f"✓ create_propositional_symbol('{predicate}', {args})")
            print(f"  → {result:40}")
        else:
            print(f"✗ create_propositional_symbol('{predicate}', {args})")
            print(f"  → {result:40} (expected: {expected})")
            all_passed = False

    print()
    return all_passed


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "SYMBOL NORMALIZER TEST SUITE" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    results = {
        "Hyphen Encoding/Decoding": test_hyphen_encoding_decoding(),
        "Propositional Symbol with Hyphens": test_propositional_symbol_with_hyphens(),
        "Formula Normalization with Hyphens": test_formula_normalization_with_hyphens(),
        "Symbol Restoration": test_symbol_restoration(),
        "PredicateToProposition Integration": test_integration_with_predicate_to_proposition(),
        "Grounding Map Integration": test_grounding_map_integration(),
    }

    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:50} {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
