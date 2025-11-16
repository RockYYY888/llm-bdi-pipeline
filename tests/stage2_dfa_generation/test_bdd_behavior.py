#!/usr/bin/env python3
"""
Deep dive into dd.autoref BDD library behavior

Goal: Understand exactly how BDD nodes work, especially:
1. How are variables represented?
2. How does negation work?
3. What is the structure of BDD nodes?
4. How to correctly traverse BDD for DFA construction?
"""

from dd.autoref import BDD


def test_basic_bdd_structure():
    """Test basic BDD structure for single variable"""
    print("=" * 80)
    print("TEST 1: Basic BDD Structure")
    print("=" * 80)

    bdd = BDD()
    bdd.add_var('x')

    # Create BDD for variable x
    x = bdd.var('x')

    print("\nBDD for 'x':")
    print(f"  Node: {x}")
    print(f"  Level: {x.level}")
    print(f"  Variable: {bdd.vars[x.level] if x.level is not None else 'None'}")
    print(f"  High branch: {x.high}")
    print(f"  Low branch: {x.low}")
    print(f"  bdd.true: {bdd.true}")
    print(f"  bdd.false: {bdd.false}")

    # Test truth values
    print("\nTruth table for 'x':")
    print(f"  x=True: {x.high} (should be TRUE)")
    print(f"  x=False: {x.low} (should be FALSE)")

    # Create BDD for NOT x
    not_x = ~x

    print("\nBDD for '~x':")
    print(f"  Node: {not_x}")
    print(f"  Level: {not_x.level}")
    print(f"  High branch: {not_x.high}")
    print(f"  Low branch: {not_x.low}")

    print("\nTruth table for '~x':")
    print(f"  x=True: {not_x.high} (should be FALSE)")
    print(f"  x=False: {not_x.low} (should be TRUE)")

    # Check if they're the same object
    print(f"\nx and ~x are same object: {x is not_x}")
    print(f"x == ~x: {x == not_x}")


def test_bdd_evaluation():
    """Test BDD evaluation with actual values"""
    print("\n" + "=" * 80)
    print("TEST 2: BDD Evaluation")
    print("=" * 80)

    bdd = BDD()
    bdd.add_var('x')

    x = bdd.var('x')
    not_x = ~x

    # Evaluate using BDD's built-in method
    print("\nEvaluating 'x':")
    try:
        # Test different evaluation methods
        print(f"  With x=True: {bdd.let({'x': True}, x)}")
        print(f"  With x=False: {bdd.let({'x': False}, x)}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nEvaluating '~x':")
    try:
        print(f"  With x=True: {bdd.let({'x': True}, not_x)}")
        print(f"  With x=False: {bdd.let({'x': False}, not_x)}")
    except Exception as e:
        print(f"  Error: {e}")


def test_complex_formula():
    """Test complex boolean formula"""
    print("\n" + "=" * 80)
    print("TEST 3: Complex Formula - (a & b) | c")
    print("=" * 80)

    bdd = BDD()
    bdd.add_var('a')
    bdd.add_var('b')
    bdd.add_var('c')

    a = bdd.var('a')
    b = bdd.var('b')
    c = bdd.var('c')

    # Build: (a & b) | c
    formula = (a & b) | c

    print("\nFormula: (a & b) | c")
    print(f"  Root node: {formula}")
    print(f"  Root level: {formula.level}")
    print(f"  Root variable: {bdd.vars[formula.level] if formula.level is not None else 'None'}")

    def traverse_bdd(node, indent=0, visited=None):
        """Recursively traverse and print BDD structure"""
        if visited is None:
            visited = set()

        prefix = "  " * indent

        # Terminal nodes
        if node == bdd.true:
            print(f"{prefix}→ TRUE")
            return
        if node == bdd.false:
            print(f"{prefix}→ FALSE")
            return

        # Check if already visited (for node sharing)
        node_id = id(node)
        if node_id in visited:
            var_name = bdd.vars[node.level] if node.level is not None else '?'
            print(f"{prefix}→ [SHARED: {var_name} @ {node_id}]")
            return
        visited.add(node_id)

        # Internal node
        var_name = bdd.vars[node.level] if node.level is not None else '?'
        print(f"{prefix}[{var_name}] (level={node.level}, id={node_id})")

        if node.high is not None:
            print(f"{prefix}  HIGH ({var_name}=true):")
            traverse_bdd(node.high, indent+2, visited)

        if node.low is not None:
            print(f"{prefix}  LOW ({var_name}=false):")
            traverse_bdd(node.low, indent+2, visited)

    print("\nBDD Structure:")
    traverse_bdd(formula)

    # Test evaluation
    print("\nTruth table:")
    for a_val in [False, True]:
        for b_val in [False, True]:
            for c_val in [False, True]:
                assignment = {'a': a_val, 'b': b_val, 'c': c_val}
                result = bdd.let(assignment, formula)
                expected = (a_val and b_val) or c_val
                match = "✓" if (result == bdd.true) == expected else "✗"
                print(f"  a={a_val}, b={b_val}, c={c_val}: {result} (expected: {expected}) {match}")


def test_our_formula():
    """Test our actual formula: on_d_e | (clear_c & on_a_b)"""
    print("\n" + "=" * 80)
    print("TEST 4: Our Formula - on_d_e | (clear_c & on_a_b)")
    print("=" * 80)

    bdd = BDD()
    bdd.add_var('clear_c')
    bdd.add_var('on_a_b')
    bdd.add_var('on_d_e')

    clear_c = bdd.var('clear_c')
    on_a_b = bdd.var('on_a_b')
    on_d_e = bdd.var('on_d_e')

    # Build: on_d_e | (clear_c & on_a_b)
    formula = on_d_e | (clear_c & on_a_b)

    print("\nFormula: on_d_e | (clear_c & on_a_b)")

    def traverse_bdd_detailed(node, indent=0, visited=None):
        """Detailed BDD traversal with more information"""
        if visited is None:
            visited = set()

        prefix = "  " * indent

        # Terminal nodes
        if node == bdd.true:
            print(f"{prefix}→ TRUE (id={id(node)})")
            return
        if node == bdd.false:
            print(f"{prefix}→ FALSE (id={id(node)})")
            return

        # Check if already visited
        node_id = id(node)
        if node_id in visited:
            var_name = bdd.vars[node.level] if node.level is not None else '?'
            print(f"{prefix}→ [SHARED: {var_name} @ {node_id}]")
            return
        visited.add(node_id)

        # Internal node
        var_name = bdd.vars[node.level] if node.level is not None else '?'
        print(f"{prefix}[{var_name}]")
        print(f"{prefix}  (level={node.level}, id={node_id})")
        print(f"{prefix}  high={id(node.high)}, low={id(node.low)}")

        if node.high is not None:
            print(f"{prefix}  HIGH ({var_name}=true):")
            traverse_bdd_detailed(node.high, indent+2, visited)

        if node.low is not None:
            print(f"{prefix}  LOW ({var_name}=false):")
            traverse_bdd_detailed(node.low, indent+2, visited)

    print("\nDetailed BDD Structure:")
    traverse_bdd_detailed(formula)

    # Truth table
    print("\nTruth table (showing all combinations):")
    print("  clear_c | on_a_b | on_d_e | Result | Expected")
    print("  --------|--------|--------|--------|----------")

    for clear_c_val in [False, True]:
        for on_a_b_val in [False, True]:
            for on_d_e_val in [False, True]:
                assignment = {
                    'clear_c': clear_c_val,
                    'on_a_b': on_a_b_val,
                    'on_d_e': on_d_e_val
                }
                result = bdd.let(assignment, formula)
                expected = on_d_e_val or (clear_c_val and on_a_b_val)
                result_bool = (result == bdd.true)
                match = "✓" if result_bool == expected else "✗"

                print(f"  {str(clear_c_val):7} | {str(on_a_b_val):6} | {str(on_d_e_val):6} | "
                      f"{str(result_bool):6} | {str(expected):8} {match}")


def test_negation_formula():
    """Test formula with negation: ~on_a_b"""
    print("\n" + "=" * 80)
    print("TEST 5: Negation - ~on_a_b")
    print("=" * 80)

    bdd = BDD()
    bdd.add_var('on_a_b')

    on_a_b = bdd.var('on_a_b')
    not_on_a_b = ~on_a_b

    print("\nFormula: ~on_a_b")
    print(f"  on_a_b node: {on_a_b} (id={id(on_a_b)})")
    print(f"  ~on_a_b node: {not_on_a_b} (id={id(not_on_a_b)})")
    print(f"  Same object? {on_a_b is not_on_a_b}")

    def show_node_details(node, name):
        print(f"\n{name}:")
        print(f"  node: {node}")
        print(f"  level: {node.level}")
        print(f"  high: {node.high} (id={id(node.high)})")
        print(f"  low: {node.low} (id={id(node.low)})")

        # Check if high/low are terminal
        if hasattr(node.high, 'level'):
            print(f"  high.level: {node.high.level}")
        else:
            print(f"  high is terminal")

        if hasattr(node.low, 'level'):
            print(f"  low.level: {node.low.level}")
        else:
            print(f"  low is terminal")

    show_node_details(on_a_b, "on_a_b")
    show_node_details(not_on_a_b, "~on_a_b")

    # Evaluate both
    print("\nEvaluation:")
    for val in [False, True]:
        result_pos = bdd.let({'on_a_b': val}, on_a_b)
        result_neg = bdd.let({'on_a_b': val}, not_on_a_b)
        print(f"  on_a_b={val}:")
        print(f"    on_a_b evaluates to: {result_pos} (is_true: {result_pos == bdd.true})")
        print(f"    ~on_a_b evaluates to: {result_neg} (is_true: {result_neg == bdd.true})")


def run_all_tests():
    """Run all BDD behavior tests"""
    tests = [
        test_basic_bdd_structure,
        test_bdd_evaluation,
        test_complex_formula,
        test_our_formula,
        test_negation_formula,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
