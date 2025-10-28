#!/bin/bash
# Comprehensive test and fix script
# This will run in background to identify and document all issues

echo "=== LTL-BDI Pipeline Comprehensive Test & Fix ===" > test_results.log
echo "Started: $(date)" >> test_results.log
echo "" >> test_results.log

# Test 1: Different blocksworld scenarios
echo "TEST 1: Simple stacking (C on B)" >> test_results.log
python src/main.py "Stack block C on block B" >> test_results.log 2>&1
echo "Exit code: $?" >> test_results.log
echo "" >> test_results.log

echo "TEST 2: Three-block tower (A on B on C)" >> test_results.log
python src/main.py "Build a tower with A on top of B on top of C" >> test_results.log 2>&1
echo "Exit code: $?" >> test_results.log
echo "" >> test_results.log

echo "TEST 3: Clear block A" >> test_results.log
python src/main.py "Make sure block A is clear" >> test_results.log 2>&1
echo "Exit code: $?" >> test_results.log
echo "" >> test_results.log

echo "TEST 4: Move block from one to another" >> test_results.log
python src/main.py "Move block A from block B to block C" >> test_results.log 2>&1
echo "Exit code: $?" >> test_results.log
echo "" >> test_results.log

echo "=== Test completed: $(date) ===" >> test_results.log
