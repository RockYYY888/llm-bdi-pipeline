"""
Main Entry Point - Dual-Branch Comparison

This is the main entry point for the LTL-BDI pipeline.

Usage:
    python src/main.py "Stack block C on block B" [--mode MODE]

Modes:
    both            - Run both branches and compare (default)
    llm_agentspeak  - Run only LLM AgentSpeak baseline branch
    fond            - Run only FOND planning branch
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from dual_branch_pipeline import DualBranchPipeline


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LTL-BDI Pipeline - Dual-Branch Comparison (LLM AgentSpeak vs FOND)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/main.py "Stack block C on block B" --mode llm_agentspeak
  python src/main.py "Stack block C on block B" --mode fond
        '''
    )
    parser.add_argument('instruction', help='Natural language instruction')
    parser.add_argument('--mode', choices=['llm_agentspeak', 'fond'], default='fond',
                        help='Execution mode: llm_agentspeak or fond (default)')

    args = parser.parse_args()
    nl_instruction = args.instruction
    mode = args.mode

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
    results = pipeline.execute(nl_instruction, mode=mode)

    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    main()
