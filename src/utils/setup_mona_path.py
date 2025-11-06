"""
MONA Path Setup Utility

Automatically adds MONA binary to PATH for ltlf2dfa to work correctly.
Import this module at the beginning of any script that uses ltlf2dfa.

Usage:
    from utils.setup_mona_path import setup_mona
    setup_mona()  # Call before using ltlf2dfa
"""

import os
import sys
from pathlib import Path


def setup_mona(verbose=False):
    """
    Add MONA binary directory to PATH if not already present.

    Args:
        verbose: If True, print status messages

    Returns:
        bool: True if MONA is available, False otherwise
    """
    # Get project root (3 levels up from this file: utils/ -> src/ -> project)
    project_root = Path(__file__).parent.parent.parent.absolute()
    mona_bin_dir = project_root / "src" / "external" / "mona-1.4" / "mona-install" / "bin"

    # Check if MONA binary exists
    mona_binary = mona_bin_dir / "mona"
    if not mona_binary.exists():
        if verbose:
            print(f"WARNING: MONA binary not found at {mona_binary}")
            print("Please compile MONA first:")
            print("  cd src/external/mona-1.4")
            print("  ./configure --prefix=$(pwd)/mona-install --disable-shared --enable-static")
            print("  make && make install-strip")
        return False

    # Add to PATH if not already there
    mona_bin_str = str(mona_bin_dir)
    current_path = os.environ.get("PATH", "")

    if mona_bin_str not in current_path:
        os.environ["PATH"] = f"{mona_bin_str}:{current_path}"
        if verbose:
            print(f"✓ Added MONA to PATH: {mona_bin_dir}")
    else:
        if verbose:
            print(f"✓ MONA already in PATH")

    return True


def check_mona_available():
    """
    Check if MONA is available in PATH.

    Returns:
        bool: True if 'mona' command is available
    """
    import shutil
    return shutil.which("mona") is not None


# Auto-setup on import (can be disabled by setting environment variable)
if os.environ.get("SKIP_MONA_SETUP") != "1":
    setup_mona(verbose=False)


if __name__ == "__main__":
    # Test the setup
    print("=" * 60)
    print("MONA PATH SETUP TEST")
    print("=" * 60)

    success = setup_mona(verbose=True)

    if success and check_mona_available():
        print("\n✓ MONA is properly configured and available")

        # Try to get MONA version
        import subprocess
        try:
            result = subprocess.run(
                ["mona", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"\nMONA version info:\n{result.stderr[:200]}")
        except Exception as e:
            print(f"\nWarning: Could not get MONA version: {e}")
    else:
        print("\n✗ MONA is not available - ltlf2dfa will not work correctly")
        sys.exit(1)
