"""
Tests package initialization

Adds src directory to Python path for imports
"""

import sys
from pathlib import Path

# Add src to path so tests can import modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
