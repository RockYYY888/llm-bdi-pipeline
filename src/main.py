"""
Main Entry Point - Dual-Branch MVP Demo

This is the main entry point for the LTL-BDI pipeline.

Usage:
    python src/main.py "Stack block C on block B"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from dual_branch_pipeline import DualBranchPipeline


def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python src/main.py \"<natural language instruction>\"")
        print("\nExample:")
        print('  python src/main.py "Stack block C on block B"')
        print("\nMake sure to:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        sys.exit(1)

    nl_instruction = sys.argv[1]

    # Validate configuration
    config = get_config()
    if not config.validate():
        print("="*80)
        print("ERROR: OpenAI API Key Not Configured")
        print("="*80)
        print("\nThe LTL-BDI pipeline requires an OpenAI API key to function.")
        print("\nPlease follow these steps:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Edit .env and add your API key:")
        print("   OPENAI_API_KEY=sk-proj-your-actual-key-here")
        print("\n3. Run the pipeline again")
        print("\n" + "="*80)
        sys.exit(1)

    # Execute pipeline
    pipeline = DualBranchPipeline()
    results = pipeline.execute(nl_instruction)

    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    main()
