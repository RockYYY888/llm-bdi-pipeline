"""
Main entry point for the LTLf -> DFA -> HTN synthesis -> PANDA -> AgentSpeak -> Jason pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add src to path (only once)
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import get_config
from ltl_bdi_pipeline import LTL_BDI_Pipeline


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LTL-BDI Pipeline - DFA-AgentSpeak Generation from Natural Language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python src/main.py "Stack block C on block B" --domain-file ./src/domains/blocksworld/domain.hddl
  python src/main.py "Stack block C on block B" --domain-file ./src/domains/blocksworld/domain.hddl --mode dfa_agentspeak
  python src/main.py "Given blocks a, b, c on table, stack a on b on c" --domain-file ./src/domains/blocksworld/domain.hddl

Note:
  Only 'dfa_agentspeak' mode is supported.
  Domain file is mandatory and must be provided with --domain-file.
  The pipeline runs Stage 3 (HTN synthesis), Stage 4 (PANDA planning),
  Stage 5 (AgentSpeak rendering), and Stage 6 (Jason runtime validation).
        '''
    )
    parser.add_argument('instruction', help='Natural language instruction')
    parser.add_argument(
        '--domain-file',
        required=True,
        help='Path to HDDL domain file (required)',
    )
    parser.add_argument(
        '--mode',
        choices=['dfa_agentspeak'],
        default='dfa_agentspeak',
        help='Execution mode (default: dfa_agentspeak)'
    )

    args = parser.parse_args()
    nl_instruction = args.instruction
    mode = args.mode
    domain_file = str(Path(args.domain_file).expanduser().resolve())

    if not Path(domain_file).exists():
        print("="*80)
        print("ERROR: Domain File Not Found")
        print("="*80)
        print(f"\nProvided --domain-file path does not exist:\n{domain_file}")
        sys.exit(1)

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
        print("   OPENAI_MODEL=deepseek-chat  # Default model")
        print("\n3. Run the pipeline again")
        print("\n" + "="*80)
        sys.exit(1)

    # Execute pipeline
    pipeline = LTL_BDI_Pipeline(domain_file=domain_file)
    results = pipeline.execute(nl_instruction, mode=mode)

    # Exit with appropriate code
    sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
    main()
