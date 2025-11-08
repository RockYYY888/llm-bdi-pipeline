#!/bin/bash
# Wrapper script to run Jason with correct Java version

# Set Java 23 (Corretto)
export JAVA_HOME=/Users/lyw/Library/Java/JavaVirtualMachines/corretto-23.0.2/Contents/Home

# Set Jason home
export JASON_HOME=/Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev/src/stage4_jason_validation/jason_src
export PATH=$JASON_HOME/bin:$PATH

# Run Jason with the provided .mas2j file
jason "$@"
