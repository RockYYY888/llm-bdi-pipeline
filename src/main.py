"""
Main Entry Point - LTL-BDI Pipeline

This is the main entry point for the LTL-BDI pipeline (DFA-AgentSpeak Generation).

Usage:
    python src/main.py "Stack block C on block B"
    python src/main.py "Stack block C on block B" --mode dfa_agentspeak

Note: FOND planning mode has been moved to src/legacy/fond/
"""

import sys
import argparse
from pathlib import Path

# Add src to path (only once)
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config import get_config
from ltl_bdi_pipeline import LTL_BDI_Pipeline


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LTL-BDI Pipeline - DFA-AgentSpeak Generation from Natural Language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/main.py "Stack block C on block B"
  python src/main.py "Stack block C on block B" --mode dfa_agentspeak
  python src/main.py "Given blocks a, b, c on table, stack a on b on c"

Note:
  Only 'dfa_agentspeak' mode is supported (default)
  FOND planning mode has been moved to src/legacy/fond/
  See src/legacy/fond/README.md for restoration instructions
        '''
    )
    parser.add_argument('instruction', help='Natural language instruction')
    parser.add_argument(
        '--mode',
        choices=['dfa_agentspeak'],
        default='dfa_agentspeak',
        help='Execution mode (default: dfa_agentspeak)'
    )

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
        print("   OPENAI_MODEL=deepseek-chat  # or gpt-4o-mini")
        print("\n3. Run the pipeline again")
        print("\n" + "="*80)
        sys.exit(1)

    # Execute pipeline
    pipeline = LTL_BDI_Pipeline()
    results = pipeline.execute(nl_instruction, mode=mode)

    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    main()
