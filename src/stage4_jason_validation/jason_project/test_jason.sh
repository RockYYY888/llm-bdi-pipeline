#!/bin/bash
# Test script for Jason AgentSpeak validation

echo "========================================"
echo "Stage 4: Jason AgentSpeak Validation"
echo "========================================"
echo ""
echo "Testing Jason BDI Framework..."
echo ""

# Set Java 23 (required for Jason)
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home

# Run Jason with the simple test agent
$JAVA_HOME/bin/java -jar ../jason_src/jason-cli/build/bin/jason-cli-all-3.3.1.jar simple_test.mas2j --console

echo ""
echo "========================================"
echo "Test completed"
echo "========================================"
