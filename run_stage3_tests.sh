#!/bin/bash
# Run all Stage 3 backward planning tests
#
# Usage: ./run_stage3_tests.sh [test_name]
#   test_name: optional, specify 'integration' or 'diagnostic'
#   If not specified, runs all tests

set -e  # Exit on error

echo "========================================================================"
echo "Stage 3 Backward Planning Test Suite"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2

    echo "------------------------------------------------------------------------"
    echo "Running: $test_name"
    echo "------------------------------------------------------------------------"

    if python "$test_file"; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
        return 1
    fi
}

# Track results
total_tests=0
passed_tests=0
failed_tests=0

# Determine which tests to run
test_type=${1:-all}

case $test_type in
    integration)
        echo "Running integration tests only..."
        echo ""

        run_test "Integration Tests" "tests/stage3_code_generation/test_integration_backward_planner.py"
        exit_code=$?
        total_tests=1
        if [ $exit_code -eq 0 ]; then
            passed_tests=1
        else
            failed_tests=1
        fi
        ;;

    diagnostic)
        echo "Running diagnostic test only..."
        echo ""

        run_test "Diagnostic Test" "tests/stage3_code_generation/test_diagnostic_plans.py"
        exit_code=$?
        total_tests=1
        if [ $exit_code -eq 0 ]; then
            passed_tests=1
        else
            failed_tests=1
        fi
        ;;

    all|*)
        echo "Running all tests..."
        echo ""

        # Run integration tests
        run_test "Integration Tests" "tests/stage3_code_generation/test_integration_backward_planner.py"
        if [ $? -eq 0 ]; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
        ((total_tests++))

        echo ""

        # Run diagnostic test
        run_test "Diagnostic Test" "tests/stage3_code_generation/test_diagnostic_plans.py"
        if [ $? -eq 0 ]; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
        ((total_tests++))
        ;;
esac

# Summary
echo ""
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo "Total tests run: $total_tests"
echo -e "Passed: ${GREEN}$passed_tests${NC}"
echo -e "Failed: ${RED}$failed_tests${NC}"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
fi
