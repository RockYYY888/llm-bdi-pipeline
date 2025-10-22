"""
Main Entry Point

This is the main entry point for the LTL-BDI pipeline.
It provides a command-line interface to run the complete workflow:
    Stage 1: Natural Language -> LTL Specification
    Stage 2: LTL Specification -> PDDL Problem
    Stage 3: PDDL Problem -> Action Plan

Usage:
    python src/main.py "Put block A on block B"

Note: The actual orchestration logic is in orchestrator.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from orchestrator import PipelineOrchestrator


def main():
    """Main entry point - orchestrates the workflow"""
    if len(sys.argv) < 2:
        print("Usage: python src/main.py \"<natural language instruction>\"")
        print("\nExample:")
        print('  python src/main.py "Put block A on block B"')
        print("\nMake sure to:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        sys.exit(1)

    nl_instruction = sys.argv[1]

    # Check config
    config = get_config()
    if not config.validate():
        print("="*80)
        print("ERROR: OpenAI API Key Not Configured")
        print("="*80)
        print("\nPlease follow these steps:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Edit .env and add your API key:")
        print("   OPENAI_API_KEY=sk-proj-your-actual-key-here")
        print("\n3. Run the pipeline again")
        print("\nNote: Without API key, mock parser will be used (limited functionality)")
        print("="*80)

        # Continue with mock parser
        print("\nContinuing with MOCK parser (no LLM API calls)...\n")

    # Initialize orchestrator and run workflow
    orchestrator = PipelineOrchestrator()
    orchestrator.execute(nl_instruction)


if __name__ == "__main__":
    main()
