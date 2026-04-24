"""
Command-line entry point for the Chapter 4 aligned plan-library pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from evaluation import PlanLibraryEvaluationPipeline
from plan_library import PlanLibraryGenerationPipeline
from utils.config import get_config


def _absolute_path(path_text: str | None) -> str | None:
	if not path_text:
		return None
	return str(Path(path_text).expanduser().resolve())


def _require_existing_path(path_text: str | None, *, label: str) -> str:
	resolved_path = _absolute_path(path_text)
	if not resolved_path or not Path(resolved_path).exists():
		print("=" * 80)
		print(f"ERROR: {label} Not Found")
		print("=" * 80)
		print(f"\nProvided path does not exist:\n{resolved_path}")
		sys.exit(1)
	return resolved_path


def _has_configured_api_key(api_key: str | None) -> bool:
	return bool(api_key and api_key.startswith("sk-"))


def _require_api_key(
	*,
	api_key: str | None,
	env_var_name: str,
	config,
	purpose: str,
	fallback_env_var_name: str | None = None,
) -> None:
	if _has_configured_api_key(api_key):
		return
	print("=" * 80)
	print(f"ERROR: {env_var_name} Not Configured")
	print("=" * 80)
	print(f"\n{purpose} requires an OpenAI API key.")
	print("\nPlease follow these steps:")
	print("1. Copy .env.example to .env:")
	print("   cp .env.example .env")
	print("\n2. Edit .env and add your API key:")
	print(f"   {env_var_name}=sk-proj-your-actual-key-here")
	if fallback_env_var_name:
		print(f"   # fallback: {fallback_env_var_name}=sk-proj-your-actual-key-here")
	print(
		f"   GOAL_GROUNDING_MODEL={config.goal_grounding_model}  "
		"# temporal specification grounding default",
	)
	print(
		f"   METHOD_SYNTHESIS_MODEL={config.method_synthesis_model}  "
		"# method library synthesis default",
	)
	print("\n3. Run the command again")
	print("\n" + "=" * 80)
	sys.exit(1)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Generate and evaluate Chapter 4 plan-library artifacts following the paper pipeline "
			"D^- + L_s -> Φ_s -> M -> S."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python src/main.py generate-library --domain-file ./src/domains/blocksworld/domain.hddl
  python src/main.py evaluate-library --library-artifact ./artifacts/plan_library/blocksworld --domain-file ./src/domains/blocksworld/domain.hddl --query-id query_1
  python src/main.py evaluate-library --library-artifact ./artifacts/plan_library/blocksworld --domain-file ./src/domains/blocksworld/domain.hddl --problem-file ./src/domains/blocksworld/problems/p01.hddl --instruction "Put block b4 on block b2" --ltlf-formula "do_put_on(b4, b2)"
		""",
	)
	subparsers = parser.add_subparsers(dest="command")

	generate_parser = subparsers.add_parser(
		"generate-library",
		help="Generate the Chapter 4 method library M and AgentSpeak(L) plan library S.",
	)
	generate_parser.add_argument("--domain-file", required=True, help="Path to the HDDL domain file")
	generate_parser.add_argument(
		"--query-dataset",
		help="Optional path to a stored temporal-specification dataset. Defaults to queries_LTLf.json.",
	)
	generate_parser.add_argument(
		"--query-domain",
		help="Optional explicit dataset domain key. Otherwise inferred from the domain file.",
	)
	generate_parser.add_argument(
		"--query-id",
		action="append",
		help=(
			"Stored benchmark query identifier to include in generation. "
			"Repeat to generate from multiple selected queries. Defaults to all domain queries."
		),
	)
	generate_parser.add_argument(
		"--output-root",
		help="Optional explicit output root for the persisted plan-library artifact bundle.",
	)

	evaluate_parser = subparsers.add_parser(
		"evaluate-library",
		help="Evaluate a stored plan-library bundle against a benchmark case or ad hoc instruction.",
	)
	evaluate_parser.add_argument(
		"--library-artifact",
		required=True,
		help="Path to a persisted plan-library artifact directory or one of its JSON files.",
	)
	evaluate_parser.add_argument("--domain-file", required=True, help="Path to the HDDL domain file")
	evaluate_parser.add_argument(
		"--query-id",
		help="Stored benchmark query identifier from queries_LTLf.json.",
	)
	evaluate_parser.add_argument(
		"--query-dataset",
		help="Optional path to a stored temporal-specification dataset. Defaults to queries_LTLf.json.",
	)
	evaluate_parser.add_argument(
		"--query-domain",
		help="Optional explicit dataset domain key. Otherwise inferred from the domain file.",
	)
	evaluate_parser.add_argument(
		"--problem-file",
		help="Explicit HDDL problem file for ad hoc evaluation.",
	)
	evaluate_parser.add_argument(
		"--instruction",
		help="Natural-language instruction for ad hoc evaluation.",
	)
	evaluate_parser.add_argument(
		"--ltlf-formula",
		help="Optional explicit LTLf formula. If omitted, live grounding is used for the ad hoc instruction.",
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()
	if not args.command:
		parser.print_help()
		sys.exit(2)

	config = get_config()

	if args.command == "generate-library":
		domain_file = _require_existing_path(args.domain_file, label="Domain File")
		_require_api_key(
			api_key=config.method_synthesis_api_key,
			env_var_name="METHOD_SYNTHESIS_OPENAI_API_KEY",
			config=config,
			purpose="Plan-library generation",
			fallback_env_var_name="OPENAI_API_KEY",
		)
		pipeline = PlanLibraryGenerationPipeline(
			domain_file=domain_file,
			query_dataset=_absolute_path(args.query_dataset),
			query_domain=args.query_domain,
			query_ids=tuple(args.query_id or ()),
		)
		results = pipeline.build_library_bundle(output_root=_absolute_path(args.output_root))
	elif args.command == "evaluate-library":
		domain_file = _require_existing_path(args.domain_file, label="Domain File")
		library_artifact = _require_existing_path(args.library_artifact, label="Library Artifact")
		pipeline = PlanLibraryEvaluationPipeline(domain_file=domain_file)
		if args.query_id:
			results = pipeline.evaluate_benchmark_case(
				library_artifact=library_artifact,
				query_id=args.query_id,
				query_dataset=_absolute_path(args.query_dataset),
				query_domain=args.query_domain,
			)
		else:
			if not args.problem_file or not args.instruction:
				parser.error(
					"evaluate-library requires either --query-id or the pair "
					"--problem-file and --instruction",
				)
			problem_file = _require_existing_path(args.problem_file, label="Problem File")
			if not str(args.ltlf_formula or "").strip():
				_require_api_key(
					api_key=config.openai_api_key,
					env_var_name="OPENAI_API_KEY",
					config=config,
					purpose="Ad hoc evaluation without a precomputed LTLf formula",
				)
			results = pipeline.evaluate_instruction(
				library_artifact=library_artifact,
				instruction=args.instruction,
				problem_file=problem_file,
				ltlf_formula=args.ltlf_formula,
			)
	else:
		parser.error(f"Unsupported command {args.command!r}")
		return

	print(json.dumps(results, indent=2, default=str))
	sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
	main()
