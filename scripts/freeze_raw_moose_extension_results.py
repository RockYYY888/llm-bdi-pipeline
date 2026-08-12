#!/usr/bin/env python3
"""Freeze five-seed Raw MOOSE extension results for the paper."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any
from typing import Mapping
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_aaai_comparison_tables import (  # noqa: E402
	REGISTERED_MOOSE_SCOPE_CONTRACTS,
)
from scripts.generate_aaai_comparison_tables import (  # noqa: E402
	REGISTERED_RAW_MOOSE_EXTENSION_DOMAINS,
)
from scripts.generate_aaai_comparison_tables import (  # noqa: E402
	_validate_published_moose_reference,
)
from scripts.public_result_schema import outcome_only_payload  # noqa: E402


EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_NUM_WORKERS = 6
EXPECTED_TIMEOUT_SECONDS = 1800
EXPECTED_MAX_RSS_GB = 8.0
DEFAULT_PUBLISHED_REFERENCE = (
	PROJECT_ROOT
	/ "paper_artifacts/gp2pl_evaluation/v1/moose_published_reference.json"
)
DEFAULT_OUTPUT_JSON = (
	PROJECT_ROOT
	/ "paper_artifacts/gp2pl_evaluation/v1/"
	/ "raw_moose_extension_five_seed_summary.json"
)
DEFAULT_LATEX_OUTPUT_DIR = PROJECT_ROOT / "artifacts/evaluation_tables"
DEFAULT_SUMMARY_FILES = {
	0: (
		PROJECT_ROOT
		/ "artifacts/external_planning_references/"
		/ "aaai-raw-moose-72b0604f-seed0/summary.json"
	),
	**{
		seed: (
			PROJECT_ROOT
			/ "artifacts/external_planning_references/"
			/ f"aaai-raw-moose-extension-9a0ee00d-seed{seed}/summary.json"
		)
		for seed in range(1, 5)
	},
}


def main() -> None:
	"""Validate local summaries and write portable paper result files."""

	parser = argparse.ArgumentParser(
		description="Freeze five-seed Raw MOOSE extension results.",
	)
	parser.add_argument(
		"--seed-summary",
		action="append",
		default=[],
		metavar="SEED=PATH",
		help="Repeat for seeds 0--4; defaults to the registered local runs.",
	)
	parser.add_argument(
		"--published-reference",
		type=Path,
		default=DEFAULT_PUBLISHED_REFERENCE,
	)
	parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
	parser.add_argument(
		"--latex-output-dir",
		type=Path,
		default=DEFAULT_LATEX_OUTPUT_DIR,
	)
	args = parser.parse_args()

	summary_files = (
		_parse_seed_assignments(args.seed_summary)
		if args.seed_summary
		else DEFAULT_SUMMARY_FILES
	)
	result = build_raw_moose_extension_dataset(
		summary_files,
		published_reference_file=args.published_reference,
	)
	write_result_files(
		result,
		output_json=args.output_json,
		latex_output_dir=args.latex_output_dir,
	)
	print(
		"frozen Raw MOOSE extension results "
		f"valid={result['aggregate']['pooled_valid_count']}/"
		f"{result['aggregate']['pooled_evaluation_count']} "
		f"mean={result['aggregate']['mean_valid_count']:.1f}/"
		f"{result['protocol']['case_count_per_seed']} "
		f"output={Path(args.output_json).expanduser().resolve()}",
	)


def build_raw_moose_extension_dataset(
	summary_files: Mapping[int, str | Path],
	*,
	published_reference_file: str | Path,
) -> dict[str, Any]:
	"""Build one portable dataset from five complete local Raw MOOSE runs."""

	if set(summary_files) != set(EXPECTED_SEEDS):
		raise ValueError("Raw MOOSE summaries must exactly cover seeds 0--4")
	published_path = Path(published_reference_file).expanduser().resolve()
	published = _read_json(published_path)
	_validate_published_moose_reference(published)
	expected_case_ids = _registered_extension_case_ids()
	expected_contract = REGISTERED_MOOSE_SCOPE_CONTRACTS["gp2pl_extension"]
	_validate_case_contract(expected_case_ids, expected_contract)

	seed_results: list[dict[str, Any]] = []
	portable_records: list[dict[str, Any]] = []
	domain_seed_valid: dict[str, dict[int, int]] = defaultdict(dict)
	domain_seed_total: dict[str, dict[int, int]] = defaultdict(dict)
	domain_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
	for seed in EXPECTED_SEEDS:
		summary_path = Path(summary_files[seed]).expanduser().resolve()
		summary = _read_json(summary_path)
		_validate_summary_header(summary, seed=seed)
		selected = tuple(
			record
			for record in summary.get("results") or ()
			if str(record.get("domain") or "")
			in REGISTERED_RAW_MOOSE_EXTENSION_DOMAINS
		)
		case_ids = tuple(_case_id(record) for record in selected)
		_validate_case_contract(case_ids, expected_contract)
		if set(case_ids) != expected_case_ids:
			raise ValueError(f"Raw MOOSE seed {seed} changes the extension case set")

		status_counts: Counter[str] = Counter()
		valid_count = 0
		for record in sorted(selected, key=_case_id):
			portable = _portable_record(record, seed=seed)
			portable_records.append(portable)
			status = str(portable["status"])
			status_counts[status] += 1
			domain = str(portable["domain"])
			domain_status_counts[domain][status] += 1
			domain_seed_total[domain][seed] = (
				domain_seed_total[domain].get(seed, 0) + 1
			)
			if portable["valid"] is True:
				valid_count += 1
				domain_seed_valid[domain][seed] = (
					domain_seed_valid[domain].get(seed, 0) + 1
				)

		seed_results.append(
			{
				"seed": seed,
				"source_result_count": len(summary.get("results") or ()),
				"selected_result_count": len(selected),
				"valid_count": valid_count,
				"failure_count": len(selected) - valid_count,
				"status_counts": dict(sorted(status_counts.items())),
			}
		)

	domain_rows = _aggregate_domains(
		domain_seed_valid=domain_seed_valid,
		domain_seed_total=domain_seed_total,
		domain_status_counts=domain_status_counts,
	)
	valid_counts = [int(row["valid_count"]) for row in seed_results]
	case_count = len(expected_case_ids)
	published_results = dict(published["published_results"])
	mean_valid = statistics.mean(valid_counts)
	sd_valid = statistics.stdev(valid_counts)
	return outcome_only_payload({
		"artifact_kind": "gp2pl_raw_moose_extension_five_seed_result",
		"schema_version": 1,
		"protocol": {
			"method": "Raw MOOSE",
			"source": "Measured",
			"seeds": list(EXPECTED_SEEDS),
			"domains": list(REGISTERED_RAW_MOOSE_EXTENSION_DOMAINS),
			"case_count_per_seed": case_count,
			"record_count": len(portable_records),
			"case_contract": dict(expected_contract),
			"num_workers": EXPECTED_NUM_WORKERS,
			"timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
			"max_rss_gb": EXPECTED_MAX_RSS_GB,
			"plan_verifier_timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
			"source_partition": "a_priori_disjoint_domain_scopes",
			"post_hoc_domain_source_selection": False,
			"runtime_comparison_allowed": False,
			"clean_git_state_required": False,
		},
		"published_reference": {
			"arxiv_version": str(dict(published["source"])["arxiv_version"]),
			"table": str(dict(published["source"])["table"]),
			"source": "Reported",
			"seed_count": int(published_results["seed_count"]),
			"domain_count": len(published_results["domains"]),
			"case_count_per_seed": int(
				published_results["case_count_per_seed"],
			),
			"mean_solved_count": float(published_results["mean_solved_count"]),
			"coverage_rate": (
				float(published_results["mean_solved_count"])
				/ int(published_results["case_count_per_seed"])
			),
		},
		"seed_results": seed_results,
		"aggregate": {
			"mean_valid_count": mean_valid,
			"sample_sd_valid_count": sd_valid,
			"mean_valid_rate": mean_valid / case_count,
			"sample_sd_valid_rate": sd_valid / case_count,
			"pooled_valid_count": sum(valid_counts),
			"pooled_evaluation_count": case_count * len(EXPECTED_SEEDS),
			"failure_count": case_count * len(EXPECTED_SEEDS) - sum(valid_counts),
			"status_counts": dict(
				sorted(Counter(record["status"] for record in portable_records).items()),
			),
		},
		"domains": domain_rows,
		"records": portable_records,
	})


def render_moose_reference_table(result: Mapping[str, Any]) -> str:
	"""Render the paper table separating reported and measured MOOSE evidence."""

	published = dict(result["published_reference"])
	aggregate = dict(result["aggregate"])
	protocol = dict(result["protocol"])
	lines = [
		"% Auto-generated by scripts/freeze_raw_moose_extension_results.py.",
		"\\begin{table}[htbp]",
		"\\centering",
		"\\small",
		"\\setlength{\\tabcolsep}{1.5pt}",
		"\\begin{tabular}{lrr}",
		"\\toprule",
		r"Scope & Valid/seed & Coverage (\%) \\",
		"\\midrule",
		r"\multicolumn{3}{l}{\textbf{Reported} (MOOSE Table~4)} \\",
		"Original 12 domains & "
		f"{float(published['mean_solved_count']):.1f}/"
		f"{_format_int(published['case_count_per_seed'])} & "
		f"{float(published['coverage_rate']) * 100:.2f} " + r"\\",
		"\\midrule",
		r"\multicolumn{3}{l}{\textbf{Measured} (local five seeds)} \\",
		"GP2PL-added 4 & "
		f"{float(aggregate['mean_valid_count']):.1f} $\\pm$ "
		f"{float(aggregate['sample_sd_valid_count']):.1f}/"
		f"{_format_int(protocol['case_count_per_seed'])} & "
		f"{float(aggregate['mean_valid_rate']) * 100:.2f} $\\pm$ "
		f"{float(aggregate['sample_sd_valid_rate']) * 100:.2f} " + r"\\",
	]
	for row in result["domains"]:
		lines.append(
			f"\\quad {_domain_label(str(row['domain']))} & "
			f"{float(row['mean_valid_count']):.1f} $\\pm$ "
			f"{float(row['sample_sd_valid_count']):.1f}/"
			f"{int(row['test_count'])} & "
			f"{float(row['mean_valid_rate']) * 100:.2f} $\\pm$ "
			f"{float(row['sample_sd_valid_rate']) * 100:.2f} " + r"\\",
		)
	lines.extend(
		(
			"\\bottomrule",
			"\\end{tabular}",
			"\\caption{Raw MOOSE coverage under the declared source split. "
			"The Reported row is copied from Table~4 of the five-seed MOOSE "
			"extended paper~\\cite{Chen2025MooseExtended}; the Measured rows "
			"are mean $\\pm$ sample standard deviation over the five local "
			"evidence seeds. The scopes are disjoint, and no cross-hardware runtime "
			"comparison is made.}",
			"\\label{tab:moose-reference}",
			"\\end{table}",
		),
	)
	return "\n".join(lines) + "\n"


def render_result_macros(result: Mapping[str, Any]) -> str:
	"""Render paper prose values from the frozen result."""

	seed_rows = tuple(result["seed_results"])
	aggregate = dict(result["aggregate"])
	return "\n".join(
		(
			"% Auto-generated by scripts/freeze_raw_moose_extension_results.py.",
			"\\newcommand{\\RawMooseExtensionSeedCounts}"
			f"{{{', '.join(str(row['valid_count']) for row in seed_rows)}}}",
			"\\newcommand{\\RawMooseExtensionMeanValid}"
			f"{{{float(aggregate['mean_valid_count']):.1f}}}",
			"\\newcommand{\\RawMooseExtensionSDValid}"
			f"{{{float(aggregate['sample_sd_valid_count']):.1f}}}",
			"\\newcommand{\\RawMooseExtensionMeanPercent}"
			f"{{{float(aggregate['mean_valid_rate']) * 100:.2f}}}",
			"\\newcommand{\\RawMooseExtensionSDPercent}"
			f"{{{float(aggregate['sample_sd_valid_rate']) * 100:.2f}}}",
		),
	) + "\n"


def write_result_files(
	result: Mapping[str, Any],
	*,
	output_json: str | Path,
	latex_output_dir: str | Path,
) -> None:
	"""Write deterministic JSON, table, and prose macros."""

	json_path = Path(output_json).expanduser().resolve()
	latex_root = Path(latex_output_dir).expanduser().resolve()
	json_path.parent.mkdir(parents=True, exist_ok=True)
	latex_root.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(result, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	(latex_root / "result_moose_reference_table.tex").write_text(
		render_moose_reference_table(result),
		encoding="utf-8",
	)
	(latex_root / "result_moose_reference_macros.tex").write_text(
		render_result_macros(result),
		encoding="utf-8",
	)


def _validate_summary_header(summary: Mapping[str, Any], *, seed: int) -> None:
	if summary.get("success") is not True:
		raise ValueError(f"Raw MOOSE seed {seed} did not complete")
	if int(summary.get("infrastructure_failure_count") or 0) != 0:
		raise ValueError(f"Raw MOOSE seed {seed} has infrastructure failures")
	parameters = dict(summary.get("parameters") or {})
	expected_parameters = {
		"num_workers": EXPECTED_NUM_WORKERS,
		"timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
		"plan_verifier_timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
	}
	for key, expected in expected_parameters.items():
		if int(parameters.get(key) or 0) != expected:
			raise ValueError(f"Raw MOOSE seed {seed} has unexpected {key}")
	if float(parameters.get("max_rss_gb") or 0.0) != EXPECTED_MAX_RSS_GB:
		raise ValueError(f"Raw MOOSE seed {seed} has unexpected memory limit")
	manifest = dict(summary.get("model_batch_manifest") or {})
	settings = dict(manifest.get("settings") or {})
	expected_settings = {
		"random_seed": seed,
		"num_workers": 1,
		"num_permutations": 3,
		"goal_max_size": 1,
		"train_timeout_seconds": 43200,
	}
	for key, expected in expected_settings.items():
		if int(settings.get(key) or 0) != expected:
			raise ValueError(f"Raw MOOSE seed {seed} has unexpected training {key}")
	if float(settings.get("max_rss_gb") or 0.0) != 16.0:
		raise ValueError(f"Raw MOOSE seed {seed} has unexpected training memory")


def _portable_record(record: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
	if str(record.get("method") or "") != "Raw MOOSE":
		raise ValueError(f"Raw MOOSE seed {seed} contains another method")
	if str(record.get("variant") or "") != "raw_moose":
		raise ValueError(f"Raw MOOSE seed {seed} contains another variant")
	status = str(record.get("status") or "")
	if status not in {"valid", "planner_failed", "timeout"}:
		raise ValueError(f"Raw MOOSE seed {seed} has unexpected status {status!r}")
	valid = record.get("plan_verifier_success") is True
	if valid != (status == "valid"):
		raise ValueError(f"Raw MOOSE seed {seed} has inconsistent valid status")
	return {
		"seed": seed,
		"case_id": _case_id(record),
		"domain": str(record.get("domain") or ""),
		"test": str(record.get("test") or ""),
		"status": status,
		"valid": valid,
		"planner_exit_code": record.get("planner_exit_code"),
		"action_count": int(record.get("action_count") or 0),
		"elapsed_seconds": float(record.get("elapsed_seconds") or 0.0),
		"runtime_wall_seconds": float(record.get("runtime_wall_seconds") or 0.0),
	}


def _aggregate_domains(
	*,
	domain_seed_valid: Mapping[str, Mapping[int, int]],
	domain_seed_total: Mapping[str, Mapping[int, int]],
	domain_status_counts: Mapping[str, Counter[str]],
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for domain in REGISTERED_RAW_MOOSE_EXTENSION_DOMAINS:
		totals = [int(domain_seed_total[domain].get(seed, 0)) for seed in EXPECTED_SEEDS]
		if len(set(totals)) != 1 or totals[0] <= 0:
			raise ValueError(f"Raw MOOSE domain {domain} has inconsistent totals")
		valid_counts = [
			int(domain_seed_valid[domain].get(seed, 0)) for seed in EXPECTED_SEEDS
		]
		mean_valid = statistics.mean(valid_counts)
		sd_valid = statistics.stdev(valid_counts)
		rows.append(
			{
				"domain": domain,
				"test_count": totals[0],
				"valid_counts": valid_counts,
				"mean_valid_count": mean_valid,
				"sample_sd_valid_count": sd_valid,
				"mean_valid_rate": mean_valid / totals[0],
				"sample_sd_valid_rate": sd_valid / totals[0],
				"status_counts": dict(sorted(domain_status_counts[domain].items())),
			}
		)
	return rows


def _registered_extension_case_ids() -> set[str]:
	return {
		f"{domain}:{problem_file.name}"
		for domain in REGISTERED_RAW_MOOSE_EXTENSION_DOMAINS
		for problem_file in (PROJECT_ROOT / "src/domains" / domain / "test").glob(
			"*.pddl",
		)
	}


def _case_id(record: Mapping[str, Any]) -> str:
	return (
		f"{str(record.get('domain') or '')}:"
		f"{Path(str(record.get('problem_file') or '')).name}"
	)


def _validate_case_contract(
	case_ids: Sequence[str] | set[str],
	contract: Mapping[str, Any],
) -> None:
	normalized = tuple(str(case_id) for case_id in case_ids)
	if len(normalized) != len(set(normalized)):
		raise ValueError("Raw MOOSE extension contains duplicate case identifiers")
	if len(normalized) != int(contract["count"]):
		raise ValueError("Raw MOOSE extension case contract mismatch")


def _parse_seed_assignments(values: Sequence[str]) -> dict[int, Path]:
	result: dict[int, Path] = {}
	for value in values:
		seed_text, separator, filename = str(value).partition("=")
		if not separator:
			raise ValueError(f"seed summary must use SEED=PATH: {value!r}")
		seed = int(seed_text)
		if seed in result:
			raise ValueError(f"duplicate Raw MOOSE seed {seed}")
		result[seed] = Path(filename).expanduser().resolve()
	if set(result) != set(EXPECTED_SEEDS):
		raise ValueError("Raw MOOSE summaries must exactly cover seeds 0--4")
	return result


def _read_json(path: str | Path) -> dict[str, Any]:
	return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _format_int(value: object) -> str:
	return f"{int(value):,}"


def _domain_label(domain: str) -> str:
	return domain.replace("blocksworld-", "Blocks ").replace("-", " ").title()


if __name__ == "__main__":
	main()
