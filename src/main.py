"""
Command-line entry point for the domain-complete HTN pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from offline_method_generation.pipeline import OfflineMethodGenerationPipeline
from online_query_solution.pipeline import OnlineQuerySolutionPipeline
from utils.config import get_config


def _absolute_path(path_text: str | None) -> str | None:
	if not path_text:
		return None
	return str(Path(path_text).expanduser().resolve())


def build_argument_parser() -> argparse.ArgumentParser:
	return argparse.ArgumentParser(
		description=(
			"Domain-complete HTN pipeline with masked-domain offline method generation and "
			"Jason-based online natural-language query execution."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python src/main.py --build-domain-library --domain-file ./src/domains/blocksworld/domain.hddl
  python src/main.py "Stack block C on block B" --domain-file ./src/domains/blocksworld/domain.hddl --problem-file ./src/domains/blocksworld/problems/p01.hddl
  python src/main.py "Stack block C on block B" --domain-file ./src/domains/blocksworld/domain.hddl --problem-file ./src/domains/blocksworld/problems/p01.hddl --library-artifact ./artifacts/domain_builds/blocksworld
		""",
	)


def main() -> None:
	parser = build_argument_parser()
	parser.add_argument(
		"instruction",
		nargs="?",
		help="Natural-language query to execute against a cached or freshly built domain library",
	)
	parser.add_argument(
		"--domain-file",
		required=True,
		help="Path to the HDDL domain file",
	)
	parser.add_argument(
		"--problem-file",
		help="Optional path to the HDDL problem file",
	)
	parser.add_argument(
		"--build-domain-library",
		action="store_true",
		help=(
			"Run only the offline domain-build pipeline. The build masks official methods, "
			"synthesizes one generated library for the domain, and persists masked plus "
			"generated domain artifacts."
		),
	)
	parser.add_argument(
		"--library-artifact",
		help="Path to a cached domain library directory or method_library.json file",
	)
	parser.add_argument(
		"--artifact-output-root",
		help="Optional explicit output root for persisted domain-build artifacts",
	)
	parser.add_argument(
		"--online-domain-source",
		choices=("benchmark", "generated"),
		help=(
			"Select which domain HDDL the online query path uses for grounding, "
			"AgentSpeak rendering, Jason runtime validation, and verification-domain "
			"construction. Defaults to the configured ONLINE_DOMAIN_SOURCE, currently benchmark."
		),
	)

	args = parser.parse_args()
	domain_file = _absolute_path(args.domain_file)
	problem_file = _absolute_path(args.problem_file)

	if not domain_file or not Path(domain_file).exists():
		print("=" * 80)
		print("ERROR: Domain File Not Found")
		print("=" * 80)
		print(f"\nProvided --domain-file path does not exist:\n{domain_file}")
		sys.exit(1)
	if problem_file is not None and not Path(problem_file).exists():
		print("=" * 80)
		print("ERROR: Problem File Not Found")
		print("=" * 80)
		print(f"\nProvided --problem-file path does not exist:\n{problem_file}")
		sys.exit(1)

	config = get_config()
	if not config.validate():
		print("=" * 80)
		print("ERROR: OpenAI API Key Not Configured")
		print("=" * 80)
		print("\nThe domain-complete HTN pipeline requires an OpenAI API key.")
		print("\nPlease follow these steps:")
		print("1. Copy .env.example to .env:")
		print("   cp .env.example .env")
		print("\n2. Edit .env and add your API key:")
		print("   OPENAI_API_KEY=sk-proj-your-actual-key-here")
		print(
			f"   GOAL_GROUNDING_MODEL={config.goal_grounding_model}  "
			"# goal grounding pinned default",
		)
		print(
			f"   METHOD_SYNTHESIS_MODEL={config.method_synthesis_model}  "
			"# method synthesis pinned default",
		)
		print("\n3. Run the pipeline again")
		print("\n" + "=" * 80)
		sys.exit(1)

	if args.build_domain_library:
		pipeline = OfflineMethodGenerationPipeline(domain_file=domain_file)
		results = pipeline.build_domain_library(output_root=args.artifact_output_root)
	else:
		if not args.instruction:
			parser.error("instruction is required unless --build-domain-library is used")
		pipeline = OnlineQuerySolutionPipeline(
			domain_file=domain_file,
			problem_file=problem_file,
			online_domain_source=args.online_domain_source,
		)
		if args.library_artifact:
			results = pipeline.execute_query_with_library(
				args.instruction,
				library_artifact=args.library_artifact,
			)
		else:
			results = pipeline.run_query(args.instruction)

	sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
	main()
