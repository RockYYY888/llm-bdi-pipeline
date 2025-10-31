#!/bin/bash
# Wrapper script to run Python scripts with MONA in PATH
# Usage: ./run_with_mona.sh python script.py [args...]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add MONA to PATH
export PATH="$SCRIPT_DIR/src/external/mona-1.4/mona-install/bin:$PATH"

# Verify MONA is available
if ! command -v mona &> /dev/null; then
    echo "ERROR: MONA not found. Please compile MONA first:"
    echo "  cd src/external/mona-1.4"
    echo "  ./configure --prefix=\$(pwd)/mona-install --disable-shared --enable-static"
    echo "  make && make install-strip"
    exit 1
fi

# Run the command with all arguments
"$@"
