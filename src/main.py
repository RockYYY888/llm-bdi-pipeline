"""
Command-line entry point for the dissertation plan-library workflow.
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
from evaluation.incremental_library import (
	ManualMethodPatchProvider,
	NoOpMethodPatchProvider,
	OpenAIMethodPatchProvider,
	run_incremental_jason_library_evaluation,
)
from plan_library import PlanLibraryGenerationPipeline
from temporal_specification.ltlf_dataset_generation import generate_ltlf_dataset
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
	return bool(api_key and api_key.strip())


def _require_api_key(
	*,
	api_key: str | None,
	env_var_name: str,
	config,
	purpose: str,
) -> None:
	if _has_configured_api_key(api_key):
		return
	print("=" * 80)
	print(f"ERROR: {env_var_name} Not Configured")
	print("=" * 80)
	print(f"\n{purpose} requires a language-model API key.")
	print("\nPlease follow these steps:")
	print("1. Copy .env.example to .env:")
	print("   cp .env.example .env")
	print("\n2. Edit .env and add your API key:")
	print("   LANGUAGE_MODEL_API_KEY=your-api-key")
	print(f"   # Optional stage-specific override: {env_var_name}=your-api-key")
	print(
		f"   LANGUAGE_MODEL_MODEL={config.language_model_model}  "
		"# shared language-model default",
	)
	print(
		f"   LANGUAGE_MODEL_BASE_URL={config.language_model_base_url}  "
		"# shared OpenAI-compatible endpoint",
	)
	print("\n3. Run the command again")
	print("\n" + "=" * 80)
	sys.exit(1)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Generate BDI plan libraries from masked HDDL domains and natural-language "
			"task instructions, then evaluate them with the dissertation benchmark workflow."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python src/main.py generate-ltlf-dataset --query-domain blocksworld --query-id query_1
  python src/main.py generate-library --domain-file ./src/domains/blocksworld/domain.hddl
  python src/main.py evaluate-library --library-artifact ./artifacts/plan_library/blocksworld --domain-file ./src/domains/blocksworld/domain.hddl --query-id query_1
  python src/main.py incremental-jason-evaluation --domain-file ./src/domains/blocksworld/domain.hddl --output-root ./artifacts/incremental_jason/blocksworld --query-id query_1 --patch-provider api
  python src/main.py evaluate-library --library-artifact ./artifacts/plan_library/blocksworld --domain-file ./src/domains/blocksworld/domain.hddl --problem-file ./src/domains/blocksworld/problems/p01.hddl --instruction "Put block b4 on block b2" --ltlf-formula "do_put_on(b4, b2)"
		""",
	)
	subparsers = parser.add_subparsers(dest="command")

	ltlf_parser = subparsers.add_parser(
		"generate-ltlf-dataset",
		help="Generate stored LTLf query specifications from natural-language benchmark queries.",
	)
	ltlf_parser.add_argument(
		"--source-query-dataset",
		help="Natural-language query dataset. Defaults to benchmark_queries.json.",
	)
	ltlf_parser.add_argument(
		"--output-dataset",
		help="Output LTLf dataset. Defaults to src/benchmark_data/queries_LTLf.json.",
	)
	ltlf_parser.add_argument(
		"--domain-file",
		help="Optional HDDL domain file when generating one domain.",
	)
	ltlf_parser.add_argument(
		"--query-domain",
		help="Optional query domain key. Omit to generate all supported benchmark domains.",
	)
	ltlf_parser.add_argument(
		"--query-id",
		action="append",
		help=(
			"Stored benchmark query identifier to include in LTLf generation. "
			"Repeat to generate multiple selected queries."
		),
	)
	ltlf_parser.add_argument(
		"--regenerate-existing",
		action="store_true",
		help="Regenerate records that already contain ltlf_formula.",
	)

	generate_parser = subparsers.add_parser(
		"generate-library",
		help="Generate the method library M and AgentSpeak(L) plan library S.",
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

	incremental_parser = subparsers.add_parser(
		"incremental-jason-evaluation",
		help=(
			"Incrementally construct M/S using stored LTLf queries and Jason runtime "
			"evaluation as the coverage gate."
		),
	)
	incremental_parser.add_argument("--domain-file", required=True, help="Path to the HDDL domain file")
	incremental_parser.add_argument(
		"--output-root",
		required=True,
		help="Output root for the incremental library, coverage matrix, and patch history.",
	)
	incremental_parser.add_argument(
		"--query-dataset",
		help="Optional path to a stored temporal-specification dataset. Defaults to queries_LTLf.json.",
	)
	incremental_parser.add_argument(
		"--query-domain",
		help="Optional explicit dataset domain key. Otherwise inferred from the domain file.",
	)
	incremental_parser.add_argument(
		"--query-id",
		action="append",
		help="Stored benchmark query identifier to evaluate. Repeat to run a selected subset.",
	)
	incremental_parser.add_argument(
		"--seed-artifact",
		help="Optional existing plan-library artifact to seed the incremental run.",
	)
	incremental_parser.add_argument(
		"--patch-provider",
		choices=("none", "manual", "api"),
		default="none",
		help=(
			"Patch source after a failed query. 'manual' writes offline prompts and "
			"applies matching response files; 'api' uses shared language-model config."
		),
	)
	incremental_parser.add_argument(
		"--manual-patch-dir",
		help="Directory for manual patch prompts and response files.",
	)
	incremental_parser.add_argument(
		"--max-patch-attempts",
		type=int,
		default=1,
		help="Maximum patch attempts per failed query.",
	)
	incremental_parser.add_argument(
		"--resume",
		action="store_true",
		help="Reuse successful query checkpoints and continue the incremental run.",
	)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()
	if not args.command:
		parser.print_help()
		sys.exit(2)

	config = get_config()

	if args.command == "generate-ltlf-dataset":
		domain_file = (
			_require_existing_path(args.domain_file, label="Domain File")
			if args.domain_file
			else None
		)
		_require_api_key(
			api_key=config.ltlf_generation_api_key,
			env_var_name="LTLF_GENERATION_API_KEY",
			config=config,
			purpose="LTLf dataset generation",
		)
		results = generate_ltlf_dataset(
			source_query_dataset=_absolute_path(args.source_query_dataset),
			output_dataset=_absolute_path(args.output_dataset),
			domain_file=domain_file,
			query_domain=args.query_domain,
			query_ids=tuple(args.query_id or ()),
			regenerate_existing=bool(args.regenerate_existing),
			config=config,
		)
	elif args.command == "generate-library":
		domain_file = _require_existing_path(args.domain_file, label="Domain File")
		_require_api_key(
			api_key=config.method_synthesis_api_key,
			env_var_name="METHOD_SYNTHESIS_API_KEY",
			config=config,
			purpose="Plan-library generation",
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
					api_key=config.ltlf_generation_api_key,
					env_var_name="LTLF_GENERATION_API_KEY",
					config=config,
					purpose="Ad hoc evaluation without a precomputed LTLf formula",
				)
			results = pipeline.evaluate_instruction(
				library_artifact=library_artifact,
				instruction=args.instruction,
				problem_file=problem_file,
				ltlf_formula=args.ltlf_formula,
			)
	elif args.command == "incremental-jason-evaluation":
		domain_file = _require_existing_path(args.domain_file, label="Domain File")
		if args.patch_provider == "api":
			_require_api_key(
				api_key=config.method_synthesis_api_key,
				env_var_name="METHOD_SYNTHESIS_API_KEY",
				config=config,
				purpose="Incremental method patch generation",
			)
			patch_provider = OpenAIMethodPatchProvider(config=config)
		elif args.patch_provider == "manual":
			manual_patch_dir = (
				args.manual_patch_dir
				or str(Path(args.output_root).expanduser().resolve() / "manual_patches")
			)
			patch_provider = ManualMethodPatchProvider(
				manual_dir=manual_patch_dir,
				config=config,
			)
		else:
			patch_provider = NoOpMethodPatchProvider()
		results = run_incremental_jason_library_evaluation(
			domain_file=domain_file,
			output_root=_absolute_path(args.output_root) or args.output_root,
			query_dataset=_absolute_path(args.query_dataset),
			query_domain=args.query_domain,
			query_ids=tuple(args.query_id or ()),
			seed_artifact=_absolute_path(args.seed_artifact),
			patch_provider=patch_provider,
			max_patch_attempts=int(args.max_patch_attempts),
			resume=bool(args.resume),
		)
	else:
		parser.error(f"Unsupported command {args.command!r}")
		return

	print(json.dumps(results, indent=2, default=str))
	sys.exit(0 if results.get("success", False) else 1)


if __name__ == "__main__":
	main()
