from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / "paper_artifacts/gp2pl_evaluation/v1"
PAPER_RESULT_FILES = (
	PROJECT_ROOT / "paper_artifacts/gp2pl_evaluation/v1/evaluation_results.json",
	PROJECT_ROOT / "latex_code/aamas_method_paper/figures/fig2_evaluation.diagnostic.json",
	PROJECT_ROOT / "latex_code/aamas_method_paper/figures/fig2_evaluation.metadata.json",
	PROJECT_ROOT / "latex_code/aamas_method_paper/figures/fig3_evaluation.metadata.json",
	PROJECT_ROOT / "latex_code/aamas_method_paper/figures/fig3_evaluation.diagnostic.json",
)
FORBIDDEN_EXACT_KEYS = {
	"commit",
	"compiler_freeze",
	"domain_file",
	"infrastructure_retry",
	"method_source_equivalence",
	"model_file",
	"policy_file",
	"problem_files",
	"provenance",
	"revision",
	"run_id",
	"sha256",
	"source_aggregate",
	"source_revision",
	"stderr",
	"stdout",
}
FORBIDDEN_KEY_FRAGMENTS = (
	"byte_identical",
	"fingerprint",
)
FORBIDDEN_KEY_SUFFIXES = (
	"_bytes",
	"_commit",
	"_command",
	"_dir",
	"_directory",
	"_file",
	"_files",
	"_path",
	"_paths",
	"_revision",
	"_revisions",
	"_root",
	"_run_id",
	"_sha256",
)


def test_frozen_public_results_are_outcome_only() -> None:
	files = tuple(sorted(RELEASE_ROOT.rglob("*.json"))) + PAPER_RESULT_FILES
	assert files
	for path in files:
		payload = json.loads(path.read_text(encoding="utf-8"))
		for key in _all_keys(payload):
			assert key not in FORBIDDEN_EXACT_KEYS, f"{path}: {key}"
			assert not key.endswith(FORBIDDEN_KEY_SUFFIXES), f"{path}: {key}"
			assert not any(part in key for part in FORBIDDEN_KEY_FRAGMENTS), (
				f"{path}: {key}"
			)
		serialized = json.dumps(payload, sort_keys=True)
		assert "/Users/" not in serialized
		assert '"artifacts/' not in serialized


def test_release_manifest_lists_files_without_integrity_digests() -> None:
	manifest = json.loads((RELEASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
	files = manifest["files"]
	assert isinstance(files, list)
	assert files == sorted(files)
	assert "paired_ablation_results.json" in files
	assert "benchmark_compatibility.json" not in files
	assert manifest["paired_ablation_temporal_cross_seed_record_count"] == 6140
	assert manifest["paired_ablation_temporal_cross_seed_seed_count"] == 5


def test_paired_result_aggregates_are_recomputable_from_outcomes() -> None:
	result = json.loads(
		(RELEASE_ROOT / "paired_ablation_results.json").read_text(encoding="utf-8"),
	)
	atomic_records = tuple(result["atomic_records"])
	temporal_records = tuple(result["temporal_records"])
	for row in result["atomic"]:
		records = [
			record for record in atomic_records if record["variant"] == row["variant"]
		]
		assert len(records) == row["test_count"]
		assert sum(record["valid"] is True for record in records) == row[
			"valid_trace_count"
		]
	for row in result["temporal"]:
		records = [
			record for record in temporal_records if record["variant"] == row["variant"]
		]
		assert len(records) == row["test_count"]
		assert sum(record["valid"] is True for record in records) == row[
			"valid_trace_count"
		]


def test_cross_seed_temporal_aggregate_is_recomputable_from_outcomes() -> None:
	result = json.loads(
		(RELEASE_ROOT / "paired_ablation_results.json").read_text(encoding="utf-8"),
	)
	extension = result["temporal_cross_seed"]
	records = tuple(result["temporal_cross_seed_records"])
	assert len(records) == 6140
	for row in extension["seed_results"]:
		seed_records = [record for record in records if record["seed"] == row["seed"]]
		assert len(seed_records) == row["evaluation_count"] == 1228
		assert sum(record["valid"] is True for record in seed_records) == row[
			"success_count"
		]
	assert sum(record["valid"] is True for record in records) == extension[
		"aggregate"
	]["pooled_success_count"]
	seed_zero = {
		record["sample_id"]: record for record in records if record["seed"] == 0
	}
	paired_balanced = {
		record["sample_id"]: record
		for record in result["temporal_records"]
		if record["variant"] == "certified_balanced"
	}
	assert set(seed_zero) == set(paired_balanced)
	for sample_id, cross_seed_record in seed_zero.items():
		paired_record = paired_balanced[sample_id]
		for key in (
			"action_count",
			"domain",
			"duration_seconds",
			"gold_accepted",
			"jason_status",
			"prediction_accepted",
			"profile",
			"status",
			"val_attempted",
			"val_success",
			"valid",
		):
			assert paired_record[key] == cross_seed_record[key]


def _all_keys(value: Any) -> list[str]:
	if isinstance(value, dict):
		keys: list[str] = []
		for key, item in value.items():
			keys.append(str(key))
			keys.extend(_all_keys(item))
		return keys
	if isinstance(value, list):
		return [key for item in value for key in _all_keys(item)]
	return []
