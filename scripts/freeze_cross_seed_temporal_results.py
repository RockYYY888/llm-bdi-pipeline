#!/usr/bin/env python3
"""Extend the paired experiment with five-seed Certified Balanced outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_result_schema import outcome_only_payload  # noqa: E402
from scripts.generate_aaai_comparison_tables import (  # noqa: E402
	render_comparison_macros,
)


DEFAULT_PAIRED_RESULTS = (
	PROJECT_ROOT
	/ "artifacts/paired_compiler_experiments/aaai-paired-72b0604f/paired_results.json"
)
DEFAULT_TEMPORAL_RUN_ROOT = PROJECT_ROOT / "artifacts/temporal_goal_execution_runs"
DEFAULT_RELEASE_PAIRED_RESULT = (
	PROJECT_ROOT / "paper_artifacts/gp2pl_evaluation/v1/paired_ablation_results.json"
)
DEFAULT_LATEX_MACROS_FILE = (
	PROJECT_ROOT
	/ "artifacts/evaluation_tables/result_comparison_macros.tex"
)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_CASE_COUNT = 1228
EXPECTED_DOMAINS = (
	"barman",
	"blocksworld-clear",
	"blocksworld-on",
	"blocksworld-tower",
	"depots",
	"ferry",
	"gripper",
	"logistics",
	"miconic",
	"numeric-ferry",
	"numeric-miconic",
	"numeric-minecraft",
	"numeric-transport",
	"rovers",
	"satellite",
	"transport",
)
TEMPORAL_VARIANT = "certified_balanced"
TEMPORAL_METHOD = "Certified Balanced"
REGISTERED_WORKERS = 6
REGISTERED_TIMEOUT_SECONDS = 1800
REGISTERED_JAVA_STACK_SIZE = "64m"
SHARED_SEED_ZERO_OUTCOME_FIELDS = (
	"action_count",
	"domain",
	"duration_seconds",
	"gold_accepted",
	"jason_status",
	"prediction_accepted",
	"profile",
	"sample_id",
	"status",
	"val_attempted",
	"val_success",
	"valid",
)


@dataclass(frozen=True)
class LoadedSeedRun:
	"""One normalized seed-specific temporal execution source."""

	seed: int
	batch_id: str
	benchmark_sha256: str
	parameters: Mapping[str, Any]
	results: tuple[dict[str, Any], ...]
	validated_correction_count: int


def build_cross_seed_temporal_dataset(
	summary_files: Mapping[int, str | Path],
	*,
	expected_seeds: Sequence[int] = EXPECTED_SEEDS,
	expected_domains: Sequence[str] | None = EXPECTED_DOMAINS,
	expected_case_count: int | None = EXPECTED_CASE_COUNT,
) -> dict[str, Any]:
	"""Validate five seed-bound temporal summaries and return public outcomes."""

	seed_order = tuple(int(seed) for seed in expected_seeds)
	if set(summary_files) != set(seed_order):
		raise ValueError(
			"seed summaries must exactly cover "
			f"{list(seed_order)}; observed {sorted(summary_files)}",
		)
	loaded = {
		seed: _load_seed_run(Path(summary_files[seed]), seed=seed)
		for seed in seed_order
	}
	benchmark_hashes = {run.benchmark_sha256 for run in loaded.values()}
	if len(benchmark_hashes) != 1:
		raise ValueError("cross-seed temporal summaries use different benchmarks")

	case_records: list[dict[str, Any]] = []
	case_results_by_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
	seed_results: list[dict[str, Any]] = []
	for seed in seed_order:
		run = loaded[seed]
		if expected_case_count is not None and len(run.results) != expected_case_count:
			raise ValueError(
				f"seed {seed} case count mismatch: expected {expected_case_count}, "
				f"observed {len(run.results)}",
			)
		by_case: dict[tuple[str, str], dict[str, Any]] = {}
		status_counts: Counter[str] = Counter()
		valid_count = 0
		valid_durations: list[float] = []
		action_counts: list[int] = []
		par2_values: list[float] = []
		for result in run.results:
			case_key = _case_key(result)
			if case_key in by_case:
				raise ValueError(f"seed {seed} contains duplicate case {case_key}")
			valid = _validate_case_result(result, seed=seed, case_key=case_key)
			by_case[case_key] = result
			status = str(result.get("status") or "missing")
			status_counts[status] += 1
			duration = _non_negative_seconds(result.get("duration_seconds"))
			if valid:
				valid_count += 1
				valid_durations.append(duration)
				action_counts.append(int(result["action_count"]))
				par2_values.append(duration)
			else:
				par2_values.append(float(2 * REGISTERED_TIMEOUT_SECONDS))
			case_records.append(_public_case_record(result, seed=seed, valid=valid))
		case_results_by_seed[seed] = by_case
		seed_results.append(
			{
				"seed": seed,
				"success_count": valid_count,
				"failure_count": len(by_case) - valid_count,
				"evaluation_count": len(by_case),
				"success_rate": valid_count / len(by_case),
				"status_counts": dict(sorted(status_counts.items())),
				"par2_seconds": statistics.mean(par2_values),
				"median_valid_seconds": (
					statistics.median(valid_durations) if valid_durations else None
				),
				"median_action_count": (
					statistics.median(action_counts) if action_counts else None
				),
			}
		)

	baseline_cases = set(case_results_by_seed[seed_order[0]])
	for seed in seed_order[1:]:
		if set(case_results_by_seed[seed]) != baseline_cases:
			raise ValueError(f"seed {seed} case identifiers differ from seed 0")
	observed_domains = {domain for domain, _sample_id in baseline_cases}
	if expected_domains is not None and observed_domains != set(expected_domains):
		raise ValueError(
			"cross-seed temporal domain set mismatch: "
			f"observed {sorted(observed_domains)}",
		)

	pattern_counts: Counter[str] = Counter()
	persistent_failures: list[dict[str, str]] = []
	seed_sensitive_cases: list[dict[str, str]] = []
	action_count_variant_cases: list[dict[str, Any]] = []
	action_count_invariant_count = 0
	action_count_unavailable_count = 0
	group_patterns: dict[str, dict[str, list[str]]] = {
		"domain": defaultdict(list),
		"profile": defaultdict(list),
	}
	for domain, sample_id in sorted(baseline_cases):
		results = [case_results_by_seed[seed][(domain, sample_id)] for seed in seed_order]
		validity = [_is_complete_success(result) for result in results]
		pattern = "".join("1" if valid else "0" for valid in validity)
		pattern_counts[pattern] += 1
		profile = str(results[0].get("profile") or "")
		if not profile or any(str(result.get("profile") or "") != profile for result in results):
			raise ValueError(f"profile differs across seeds for {(domain, sample_id)}")
		group_patterns["domain"][domain].append(pattern)
		group_patterns["profile"][profile].append(pattern)
		case_identity = {"domain": domain, "sample_id": sample_id}
		if pattern == "0" * len(seed_order):
			persistent_failures.append(case_identity)
		elif "0" in pattern and "1" in pattern:
			seed_sensitive_cases.append({**case_identity, "pattern": pattern})

		actions = [result.get("action_count") for result in results]
		if any(action is None for action in actions):
			action_count_unavailable_count += 1
		elif len({int(action) for action in actions}) == 1:
			action_count_invariant_count += 1
		else:
			action_count_variant_cases.append(
				{**case_identity, "action_counts": [int(action) for action in actions]},
			)

	seed_success_counts = [int(row["success_count"]) for row in seed_results]
	seed_success_rates = [float(row["success_rate"]) for row in seed_results]
	seed_par2 = [float(row["par2_seconds"]) for row in seed_results]
	pooled_evaluation_count = len(baseline_cases) * len(seed_order)
	all_success_pattern = "1" * len(seed_order)
	all_failure_pattern = "0" * len(seed_order)
	seed_sensitive_count = sum(
		count
		for pattern, count in pattern_counts.items()
		if "0" in pattern and "1" in pattern
	)

	public_result = {
		"artifact_kind": "gp2pl_cross_seed_temporal_robustness_extension",
		"schema_version": 1,
		"protocol": {
			"method": TEMPORAL_METHOD,
			"temporal_compiler_variant": TEMPORAL_VARIANT,
			"atomic_compiler_variant": "full",
			"seeds": list(seed_order),
			"domain_count": len(observed_domains),
			"case_count_per_seed": len(baseline_cases),
			"validation_workers": REGISTERED_WORKERS,
			"jason_timeout_seconds": REGISTERED_TIMEOUT_SECONDS,
			"plan_verifier_timeout_seconds": REGISTERED_TIMEOUT_SECONDS,
			"jason_java_stack_size": REGISTERED_JAVA_STACK_SIZE,
			"independent_seed_atomic_libraries": True,
			"evidence_union": False,
			"best_seed_selection": False,
			"validated_seed_zero_correction_count": sum(
				run.validated_correction_count for run in loaded.values()
			),
			"success_contract": (
				"Jason completion plus PDDL replay, neutral-goal VAL, "
				"gold-DFA acceptance, and predicted-DFA acceptance"
			),
		},
		"seed_results": seed_results,
		"aggregate": {
			"mean_success_count": statistics.mean(seed_success_counts),
			"sample_sd_success_count": statistics.stdev(seed_success_counts),
			"mean_success_rate": statistics.mean(seed_success_rates),
			"sample_sd_success_rate": statistics.stdev(seed_success_rates),
			"pooled_success_count": sum(seed_success_counts),
			"pooled_evaluation_count": pooled_evaluation_count,
			"failure_count": pooled_evaluation_count - sum(seed_success_counts),
			"all_seed_success_case_count": pattern_counts[all_success_pattern],
			"seed_sensitive_case_count": seed_sensitive_count,
			"all_seed_failure_case_count": pattern_counts[all_failure_pattern],
			"at_least_one_seed_success_case_count": (
				len(baseline_cases) - pattern_counts[all_failure_pattern]
			),
			"action_count_invariant_case_count": action_count_invariant_count,
			"action_count_variant_case_count": len(action_count_variant_cases),
			"action_count_unavailable_case_count": action_count_unavailable_count,
			"mean_seed_par2_seconds": statistics.mean(seed_par2),
			"sample_sd_seed_par2_seconds": statistics.stdev(seed_par2),
			"minimum_seed_par2_seconds": min(seed_par2),
			"maximum_seed_par2_seconds": max(seed_par2),
		},
		"domains": _group_rows(
			group_patterns["domain"],
			group_name="domain",
			seed_count=len(seed_order),
		),
		"profiles": _group_rows(
			group_patterns["profile"],
			group_name="profile",
			seed_count=len(seed_order),
		),
		"case_records": case_records,
		"case_outcomes": {
			"pattern_counts": dict(sorted(pattern_counts.items())),
			"persistent_failures": persistent_failures,
			"seed_sensitive_cases": seed_sensitive_cases,
			"action_count_variant_cases": action_count_variant_cases,
		},
	}
	return outcome_only_payload(public_result)


def merge_into_paired_result(
	result: Mapping[str, Any],
	*,
	paired_result_file: str | Path,
	update_manifest: bool,
) -> None:
	"""Insert the extension into the existing public paired result file."""

	paired_path = Path(paired_result_file).expanduser().resolve()
	paired = _read_json(paired_path)
	if paired.get("artifact_kind") != "gp2pl_paired_ablation_results":
		raise ValueError("cross-seed extension target is not the paired result file")
	case_records = list(result.get("case_records") or ())
	_reconcile_seed_zero_paired_records(paired, cross_seed_records=case_records)
	paired["temporal_cross_seed"] = {
		key: value for key, value in result.items() if key != "case_records"
	}
	paired["temporal_cross_seed_records"] = case_records
	_write_json(paired_path, paired)
	if not update_manifest:
		return
	manifest_path = paired_path.parent / "manifest.json"
	manifest = _read_json(manifest_path)
	manifest["paired_ablation_temporal_cross_seed_record_count"] = len(case_records)
	manifest["paired_ablation_temporal_cross_seed_seed_count"] = len(
		tuple(dict(result.get("protocol") or {}).get("seeds") or ()),
	)
	manifest["files"] = sorted(str(item) for item in manifest.get("files") or ())
	_write_json(manifest_path, manifest)


def write_manuscript_macros(
	*,
	paired_result_file: str | Path,
	output_file: str | Path,
) -> None:
	"""Regenerate manuscript macros from the extended paired result."""

	paired = _read_json(Path(paired_result_file).expanduser().resolve())
	text = render_comparison_macros(paired).replace(
		"% Auto-generated by scripts/generate_aaai_comparison_tables.py.",
		"% Auto-generated from "
		"paper_artifacts/gp2pl_evaluation/v1/paired_ablation_results.json.",
		1,
	)
	target = Path(output_file).expanduser().resolve()
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text(text, encoding="utf-8")


def _reconcile_seed_zero_paired_records(
	paired: dict[str, Any],
	*,
	cross_seed_records: Sequence[Mapping[str, Any]],
) -> None:
	"""Make the seed-0 extension reuse the final paired-ablation outcomes."""

	temporal_records = list(paired.get("temporal_records") or ())
	if not temporal_records:
		return
	seed_zero = {
		(str(row["domain"]), str(row["sample_id"])): row
		for row in cross_seed_records
		if int(row["seed"]) == 0
	}
	balanced_records = {
		(str(row["domain"]), str(row["sample_id"])): row
		for row in temporal_records
		if row.get("variant") == TEMPORAL_VARIANT
	}
	if set(seed_zero) != set(balanced_records):
		raise ValueError(
			"seed-0 cross-seed cases differ from Certified Balanced ablation cases",
		)
	for case, record in balanced_records.items():
		replacement = seed_zero[case]
		for field in SHARED_SEED_ZERO_OUTCOME_FIELDS:
			record[field] = replacement[field]
	balanced_aggregate = next(
		(
			row
			for row in paired.get("temporal") or ()
			if row.get("variant") == TEMPORAL_VARIANT
		),
		None,
	)
	if balanced_aggregate is None:
		raise ValueError("paired result lacks a Certified Balanced aggregate")
	balanced_aggregate["par2_seconds"] = statistics.mean(
		float(row["duration_seconds"]) if row["valid"] is True else 3_600.0
		for row in balanced_records.values()
	)


def merge_into_source_experiment(
	result: Mapping[str, Any],
	*,
	paired_results_file: str | Path,
	summary_files: Mapping[int, str | Path],
) -> None:
	"""Persist normalized seed runs inside the original paired experiment."""

	paired_path = Path(paired_results_file).expanduser().resolve()
	paired = _read_json(paired_path)
	runs: list[dict[str, Any]] = []
	for seed in EXPECTED_SEEDS:
		loaded = _load_seed_run(Path(summary_files[seed]), seed=seed)
		runs.append(
			{
				"seed": seed,
				"method": TEMPORAL_METHOD,
				"variant": TEMPORAL_VARIANT,
				"atomic_batch_id": loaded.batch_id,
				"benchmark_sha256": loaded.benchmark_sha256,
				"parameters": dict(loaded.parameters),
				"validated_correction_count": loaded.validated_correction_count,
				"results": list(loaded.results),
			}
		)
	paired["temporal_cross_seed_extension"] = {
		key: value for key, value in result.items() if key != "case_records"
	}
	paired["temporal_cross_seed_runs"] = runs
	_write_json(paired_path, paired)


def _load_seed_run(path: Path, *, seed: int) -> LoadedSeedRun:
	payload = _read_json(path.expanduser().resolve())
	if "temporal_runs" in payload:
		matches = [
			dict(run)
			for run in payload.get("temporal_runs") or ()
			if run.get("variant") == TEMPORAL_VARIANT
		]
		if len(matches) != 1:
			raise ValueError(f"seed {seed} paired result lacks one Balanced run")
		run = matches[0]
		atomic_input = dict(payload.get("temporal_atomic_input") or {})
		batch_id = str(atomic_input.get("batch_id") or "")
		corrections = sum(
			item.get("variant") == TEMPORAL_VARIANT
			for item in payload.get("incremental_result_updates") or ()
		)
	else:
		if payload.get("artifact_kind") != "temporal_goal_execution_validation":
			raise ValueError(f"seed {seed} has the wrong temporal result kind")
		if not payload.get("completed_at"):
			raise ValueError(f"seed {seed} temporal summary is incomplete")
		if payload.get("temporal_compiler_variant") != TEMPORAL_VARIANT:
			raise ValueError(f"seed {seed} does not use Certified Balanced")
		run = payload
		batch_id = Path(str(payload.get("atomic_batch_root") or "")).name
		corrections = 0
	if not re.search(rf"seed{seed}-full$", batch_id):
		raise ValueError(
			f"seed {seed} atomic batch binding is invalid: {batch_id!r}",
		)
	parameters = dict(run.get("parameters") or {})
	_validate_parameters(parameters, seed=seed)
	results = tuple(dict(item) for item in run.get("results") or ())
	benchmark_sha256 = str(run.get("benchmark_sha256") or "")
	if len(benchmark_sha256) != 64:
		raise ValueError(f"seed {seed} has no benchmark digest")
	return LoadedSeedRun(
		seed=seed,
		batch_id=batch_id,
		benchmark_sha256=benchmark_sha256,
		parameters=parameters,
		results=results,
		validated_correction_count=int(corrections),
	)


def _validate_parameters(parameters: Mapping[str, Any], *, seed: int) -> None:
	expected = {
		"jason_java_stack_size": REGISTERED_JAVA_STACK_SIZE,
		"jason_timeout_seconds": REGISTERED_TIMEOUT_SECONDS,
		"num_workers": REGISTERED_WORKERS,
		"plan_verifier_timeout_seconds": REGISTERED_TIMEOUT_SECONDS,
		"temporal_compiler_variant": TEMPORAL_VARIANT,
	}
	for key, value in expected.items():
		if parameters.get(key) != value:
			raise ValueError(
				f"seed {seed} has unexpected {key}: {parameters.get(key)!r}",
			)


def _validate_case_result(
	result: Mapping[str, Any],
	*,
	seed: int,
	case_key: tuple[str, str],
) -> bool:
	success = result.get("success")
	if not isinstance(success, bool):
		raise ValueError(f"seed {seed} case {case_key} has no Boolean outcome")
	complete_success = _is_complete_success(result)
	if success and not complete_success:
		raise ValueError(
			f"seed {seed} case {case_key} successful case lacks complete validation",
		)
	if not success and result.get("status") == "success":
		raise ValueError(f"seed {seed} case {case_key} has inconsistent status")
	return complete_success


def _is_complete_success(result: Mapping[str, Any]) -> bool:
	validation = dict(result.get("execution_validation") or {})
	action_count = result.get("action_count")
	return all(
		(
			result.get("success") is True,
			result.get("status") == "success",
			result.get("jason_status") == "success",
			isinstance(action_count, int),
			action_count is not None and action_count >= 0,
			validation.get("success") is True,
			validation.get("replay_valid") is True,
			validation.get("val_attempted") is True,
			validation.get("val_success") is True,
			validation.get("gold_accepted") is True,
			validation.get("prediction_accepted") is True,
			validation.get("action_count") == action_count,
		)
	)


def _public_case_record(
	result: Mapping[str, Any],
	*,
	seed: int,
	valid: bool,
) -> dict[str, Any]:
	validation = dict(result.get("execution_validation") or {})
	return {
		"seed": seed,
		"domain": str(result["domain"]),
		"sample_id": str(result["sample_id"]),
		"profile": str(result["profile"]),
		"status": str(result.get("status") or "missing"),
		"valid": valid,
		"jason_status": result.get("jason_status"),
		"duration_seconds": _non_negative_seconds(result.get("duration_seconds")),
		"action_count": result.get("action_count"),
		"val_attempted": validation.get("val_attempted") is True,
		"val_success": validation.get("val_success") is True,
		"gold_accepted": validation.get("gold_accepted") is True,
		"prediction_accepted": validation.get("prediction_accepted") is True,
	}


def _group_rows(
	patterns_by_group: Mapping[str, Sequence[str]],
	*,
	group_name: str,
	seed_count: int,
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for name in sorted(patterns_by_group):
		patterns = tuple(patterns_by_group[name])
		success_counts = [
			sum(pattern[index] == "1" for pattern in patterns)
			for index in range(seed_count)
		]
		rates = [count / len(patterns) for count in success_counts]
		rows.append(
			{
				group_name: name,
				"case_count_per_seed": len(patterns),
				"success_counts": success_counts,
				"mean_success_rate": statistics.mean(rates),
				"sample_sd_success_rate": statistics.stdev(rates),
				"all_seed_success_case_count": patterns.count("1" * seed_count),
				"seed_sensitive_case_count": sum(
					"0" in pattern and "1" in pattern for pattern in patterns
				),
				"all_seed_failure_case_count": patterns.count("0" * seed_count),
			}
		)
	return rows


def _case_key(result: Mapping[str, Any]) -> tuple[str, str]:
	domain = str(result.get("domain") or "").strip()
	sample_id = str(result.get("sample_id") or "").strip()
	if not domain or not sample_id:
		raise ValueError("temporal result is missing its domain or sample identifier")
	return domain, sample_id


def _non_negative_seconds(value: Any) -> float:
	seconds = float(value or 0.0)
	if seconds < 0.0:
		raise ValueError("duration_seconds must be non-negative")
	return seconds


def _default_summary_files() -> dict[int, Path]:
	return {
		0: DEFAULT_PAIRED_RESULTS,
		**{
			seed: (
				DEFAULT_TEMPORAL_RUN_ROOT
				/ f"aaai-temporal-cross-seed-balanced-seed{seed}/summary.json"
			)
			for seed in range(1, 5)
		},
	}


def _parse_seed_summaries(values: Sequence[str]) -> dict[int, Path]:
	result: dict[int, Path] = {}
	for value in values:
		seed_text, separator, path_text = value.partition("=")
		if not separator or not seed_text.isdigit() or not path_text:
			raise ValueError(f"invalid --seed-summary value: {value!r}")
		seed = int(seed_text)
		if seed in result:
			raise ValueError(f"duplicate --seed-summary for seed {seed}")
		result[seed] = Path(path_text)
	return result


def _read_json(path: Path) -> dict[str, Any]:
	if not path.is_file():
		raise ValueError(f"required result does not exist: {path}")
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError(f"expected a JSON object: {path}")
	return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--seed-summary",
		action="append",
		default=[],
		metavar="SEED=PATH",
	)
	parser.add_argument("--paired-results", type=Path, default=DEFAULT_PAIRED_RESULTS)
	parser.add_argument(
		"--release-paired-result",
		type=Path,
		default=DEFAULT_RELEASE_PAIRED_RESULT,
	)
	parser.add_argument(
		"--latex-macros-file",
		type=Path,
		default=DEFAULT_LATEX_MACROS_FILE,
	)
	parser.add_argument("--no-update-manifest", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	summary_files = (
		_parse_seed_summaries(args.seed_summary)
		if args.seed_summary
		else _default_summary_files()
	)
	result = build_cross_seed_temporal_dataset(summary_files)
	merge_into_source_experiment(
		result,
		paired_results_file=args.paired_results,
		summary_files=summary_files,
	)
	merge_into_paired_result(
		result,
		paired_result_file=args.release_paired_result,
		update_manifest=not args.no_update_manifest,
	)
	write_manuscript_macros(
		paired_result_file=args.release_paired_result,
		output_file=args.latex_macros_file,
	)
	aggregate = dict(result["aggregate"])
	print(
		"extended paired temporal experiment "
		f"valid={aggregate['pooled_success_count']}/"
		f"{aggregate['pooled_evaluation_count']} "
		f"mean={aggregate['mean_success_rate'] * 100:.2f}% "
		f"sd={aggregate['sample_sd_success_rate'] * 100:.2f}pp "
		f"output={Path(args.release_paired_result).expanduser().resolve()}",
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
