#!/bin/bash
# Wrapper script to run Jason with local stage6 toolchain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE6_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
JASON_HOME="$STAGE6_DIR/jason_src"
JASON_BIN="$JASON_HOME/bin/jason"

if [[ -n "${STAGE6_JAVA_HOME:-}" ]]; then
	export JAVA_HOME="$STAGE6_JAVA_HOME"
fi

if [[ ! -x "$JASON_BIN" ]]; then
	echo "Jason CLI launcher not found at: $JASON_BIN"
	echo "Build it first:"
	echo "  cd $JASON_HOME && ./gradlew config"
	exit 1
fi

exec "$JASON_BIN" "$@"
