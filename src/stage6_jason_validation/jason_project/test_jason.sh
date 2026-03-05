#!/bin/bash
# Test script for Jason AgentSpeak validation

set -euo pipefail

echo "========================================"
echo "Stage 6: Jason AgentSpeak Validation"
echo "========================================"
echo ""
echo "Testing Jason BDI Framework..."
echo ""

# Run Jason with the simple test agent
bash ./run_jason.sh simple_test.mas2j -v

echo ""
echo "========================================"
echo "Test completed"
echo "========================================"
