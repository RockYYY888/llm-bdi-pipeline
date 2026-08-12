#!/usr/bin/env python3
"""
Build the final result package for the current atomic-template framework.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain_level_planning.architecture_contract import (  # noqa: E402
	architecture_gap_summary,
	domain_level_architecture_contract,
)
from domain_level_planning.evidence_module import (  # noqa: E402
	compile_moose_readable_policy_to_asl_library,
)
from plan_library.rendering import render_plan_library_asl  # noqa: E402


FINAL_PAPER_MANIFEST = PROJECT_ROOT / "paper_artifacts/final_paper_manifest.json"
FINAL_RESULT_MACROS = PROJECT_ROOT / "artifacts/evaluation_tables/result_macros.tex"
FINAL_PAPER_MAIN = PROJECT_ROOT / "latex_code/aamas_method_paper/main.tex"
RESULT_MACRO_START = "% BEGIN GENERATED RESULT MACROS"
RESULT_MACRO_END = "% END GENERATED RESULT MACROS"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run the result-only data protocol for the current ASL framework.",
	)
	parser.add_argument("--output-dir", type=Path, default=Path("tmp/paper-final"))
	parser.add_argument("--manifest", type=Path, default=FINAL_PAPER_MANIFEST)
	parser.add_argument("--config-only", action="store_true")
	parser.add_argument("--validate-only", action="store_true")
	args = parser.parse_args()

	output_dir = args.output_dir.expanduser().resolve()
	manifest = load_final_paper_manifest(args.manifest)
	if args.validate_only:
		validation = validate_final_paper_package(output_dir, manifest=manifest)
		print(
			"validated final paper result package "
			f"{output_dir} checks={validation['check_count']}",
		)
		return

	_reset_managed_output(output_dir)
	write_final_paper_configs(output_dir, manifest=manifest)
	if args.config_only:
		print(f"wrote experiment configs under {output_dir}")
		return

	_write_backend_audit_logs(output_dir / "backend-audit")
	comparison = _write_current_result_only_package(output_dir=output_dir, manifest=manifest)
	validation = validate_final_paper_package(output_dir, manifest=manifest)
	print(
		"wrote final paper result package "
		f"{output_dir} checks={validation['check_count']} "
		f"reports={comparison['report_count']} baselines={comparison['baseline_count']}",
	)


def load_final_paper_manifest(
	manifest_file: str | Path = FINAL_PAPER_MANIFEST,
) -> dict[str, object]:
	"""Load the tracked final-paper result contract."""

	manifest_path = Path(manifest_file).expanduser()
	if not manifest_path.is_absolute():
		manifest_path = (PROJECT_ROOT / manifest_path).resolve()
	return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_final_paper_configs(
	output_dir: str | Path,
	*,
	manifest: dict[str, object] | None = None,
) -> dict[str, object]:
	"""Write the current empty matrix config files from the tracked manifest."""

	root = Path(output_dir).expanduser().resolve()
	root.mkdir(parents=True, exist_ok=True)
	manifest_payload = manifest or load_final_paper_manifest()
	_write_json(root / "artifact-manifest.json", manifest_payload)
	config_root = root / "configs"
	config_root.mkdir(parents=True, exist_ok=True)
	matrix_configs = dict(manifest_payload.get("matrix_configs") or {})
	paths: dict[str, str] = {}
	for name, payload in matrix_configs.items():
		config_payload = dict(payload)
		config_payload["experiments"] = []
		config_file = config_root / Path(str(config_payload.get("file") or f"{name}.json")).name
		_write_json(config_file, config_payload)
		paths[str(name)] = str(config_file)
	return {"config_paths": paths, "manifest": manifest_payload}


def validate_final_paper_package(
	output_dir: str | Path,
	*,
	manifest: dict[str, object] | None = None,
) -> dict[str, object]:
	"""Validate the current result-only final package."""

	root = Path(output_dir).expanduser().resolve()
	manifest_payload = manifest or load_final_paper_manifest()
	checks: list[str] = []

	def require(condition: bool, message: str) -> None:
		if not condition:
			raise AssertionError(message)
		checks.append(message)

	require(root.exists(), "result output directory exists")
	manifest_copy = root / "artifact-manifest.json"
	require(manifest_copy.exists(), "result manifest copy exists")
	require(
		json.loads(manifest_copy.read_text(encoding="utf-8")).get("artifact_id")
		== manifest_payload.get("artifact_id"),
		"result manifest id matches",
	)
	comparison_file = root / "comparison.json"
	require(comparison_file.exists(), "comparison file exists")
	comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
	expected = dict(manifest_payload.get("expected_package") or {})
	require(
		int(comparison.get("report_count") or 0) == int(expected.get("report_count") or 0),
		"report count matches manifest",
	)
	require(
		int(comparison.get("baseline_count") or 0) == int(expected.get("baseline_count") or 0),
		"baseline count matches manifest",
	)
	require(
		len(tuple(comparison.get("paper_table_rows") or ()))
		== int(expected.get("paper_table_row_count") or 0),
		"paper table row count matches manifest",
	)
	for name, payload in dict(manifest_payload.get("matrix_configs") or {}).items():
		config_file = root / "configs" / Path(str(dict(payload).get("file") or f"{name}.json")).name
		require(config_file.exists(), f"matrix config exists: {name}")
		config = json.loads(config_file.read_text(encoding="utf-8"))
		require(tuple(config.get("experiments") or ()) == (), f"matrix config is empty: {name}")
	require((root / "backend-audit/gp-backend-status.txt").exists(), "backend status audit exists")
	require((root / "artifact-summary.json").exists(), "result summary exists")
	summary = json.loads((root / "artifact-summary.json").read_text(encoding="utf-8"))
	require(summary.get("runtime_full_trace_planner") is False, "runtime planner disabled")
	require(
		summary.get("architecture_gap_summary")
		== architecture_gap_summary(domain_level_architecture_contract().gaps),
		"architecture contract summary is current",
	)
	_validate_atomic_smoke(root, manifest_payload, require)
	return {"check_count": len(checks), "checks": checks}


def _write_backend_audit_logs(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	commands = {
		"gp-backend-status.txt": [
			sys.executable,
			str(PROJECT_ROOT / "scripts/gp_backend_audit.py"),
			"status",
		],
		"learner-sketches-summary.txt": [
			sys.executable,
			str(PROJECT_ROOT / "scripts/gp_backend_audit.py"),
			"learner-sketches-summary",
			"--experiment",
			"all",
		],
	}
	for filename, command in commands.items():
		(output_dir / filename).write_text(_run_audit_command(command), encoding="utf-8")


def _run_audit_command(command: list[str]) -> str:
	try:
		completed = subprocess.run(
			command,
			cwd=PROJECT_ROOT,
			check=False,
			capture_output=True,
			text=True,
			timeout=60,
		)
	except Exception as error:  # noqa: BLE001 - the result record keeps diagnostics.
		return f"command_failed: {' '.join(command)}\n{error}\n"
	return (
		f"$ {' '.join(command)}\n"
		f"exit_code={completed.returncode}\n"
		f"{completed.stdout}{completed.stderr}"
	)


def _write_current_result_only_package(
	*,
	output_dir: Path,
	manifest: dict[str, object],
) -> dict[str, object]:
	smoke = dict(manifest.get("atomic_artifact_smoke") or {})
	provider_name = str(smoke.get("evidence_provider") or smoke.get("backend") or "").strip()
	domain_name = str(smoke.get("domain_name") or "unknown").strip()
	readable_policy = _resolve_project_path(
		smoke.get("readable_policy"),
		label="atomic_artifact_smoke.readable_policy",
	)
	source_name = str(smoke.get("source_name") or readable_policy.stem).replace(".model", "")
	domain_output_dir = output_dir / "atomic-artifacts" / domain_name
	domain_output_dir.mkdir(parents=True, exist_ok=True)
	if provider_name == "moose" and readable_policy.exists():
		library = compile_moose_readable_policy_to_asl_library(
			readable_policy.read_text(encoding="utf-8"),
			domain_name=domain_name,
			source_name=source_name,
			policy_file=readable_policy,
		)
		(domain_output_dir / "plan_library.json").write_text(
			json.dumps(library.to_dict(), indent=2, sort_keys=True) + "\n",
			encoding="utf-8",
		)
		(domain_output_dir / "plan_library.asl").write_text(
			render_plan_library_asl(library),
			encoding="utf-8",
		)
		atomic_status = {
			"evidence_provider": provider_name,
			"domain_name": domain_name,
			"policy_file": str(readable_policy),
			"compiled_singleton_rule_count": len(library.plans),
			"status": "compiled",
		}
	else:
		atomic_status = {
			"evidence_provider": provider_name,
			"domain_name": domain_name,
			"policy_file": str(readable_policy),
			"compiled_singleton_rule_count": 0,
			"status": "missing_or_unsupported_readable_policy",
		}
	_write_json(domain_output_dir / "atomic_library_metadata.json", atomic_status)
	contract = domain_level_architecture_contract()
	_write_json(
		output_dir / "artifact-summary.json",
		{
			"artifact_mode": "atomic_template_temporal_append",
			"architecture_gap_summary": architecture_gap_summary(contract.gaps),
			"atomic_artifact": atomic_status,
			"claim_boundary": contract.guarantee,
			"manifest_artifact_id": manifest.get("artifact_id"),
			"runtime_full_trace_planner": False,
		},
	)
	comparison = {
		"report_count": 0,
		"baseline_count": 0,
		"paper_table_rows": [],
		"artifact_mode": "atomic_template_temporal_append",
	}
	_write_json(output_dir / "comparison.json", comparison)
	return comparison


def _validate_atomic_smoke(
	output_dir: Path,
	manifest: dict[str, object],
	require: Callable[[bool, str], None],
) -> None:
	smoke = dict(manifest.get("atomic_artifact_smoke") or {})
	domain_name = str(smoke.get("domain_name") or "unknown").strip()
	metadata_file = output_dir / "atomic-artifacts" / domain_name / "atomic_library_metadata.json"
	require(metadata_file.exists(), "atomic smoke metadata exists")
	metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
	expected_provider = smoke.get("evidence_provider") or smoke.get("backend")
	require(
		metadata.get("evidence_provider") == expected_provider,
		"atomic smoke evidence provider matches",
	)
	require(
		int(metadata.get("compiled_singleton_rule_count") or 0) >= 0,
		"atomic smoke compiled rule count recorded",
	)
	asl_file = output_dir / "atomic-artifacts" / domain_name / "plan_library.asl"
	if asl_file.exists():
		asl = asl_file.read_text(encoding="utf-8")
		require("achieve_" not in asl, "atomic ASL contains no achieve_* names")
		require("transition_" not in asl, "atomic ASL contains no transition_* names")
		require("dfa_state" not in asl, "atomic ASL contains no dfa_state beliefs")


def format_comparison_latex_macros(comparison: dict[str, object]) -> str:
	"""Return the stable generated macro block for the result-only package."""

	return (
		"% Auto-generated by scripts/run_final_paper_data.py.\n"
		"% Current result-only package contains no experiment matrix rows.\n"
	)


def _write_result_macros(comparison: dict[str, object]) -> None:
	macros = format_comparison_latex_macros(comparison)
	FINAL_RESULT_MACROS.parent.mkdir(parents=True, exist_ok=True)
	FINAL_RESULT_MACROS.write_text(macros, encoding="utf-8")
	if not FINAL_PAPER_MAIN.exists():
		return
	text = FINAL_PAPER_MAIN.read_text(encoding="utf-8")
	if RESULT_MACRO_START not in text or RESULT_MACRO_END not in text:
		return
	start = text.index(RESULT_MACRO_START) + len(RESULT_MACRO_START)
	end = text.index(RESULT_MACRO_END)
	updated = f"{text[:start]}\n{macros}{text[end:]}"
	FINAL_PAPER_MAIN.write_text(updated, encoding="utf-8")


def _validate_result_macros(
	*,
	comparison: dict[str, object],
	require: Callable[[bool, str], None],
) -> None:
	expected_macros = format_comparison_latex_macros(comparison)
	require(FINAL_RESULT_MACROS.exists(), "LaTeX result macro file exists")
	require(
		FINAL_RESULT_MACROS.read_text(encoding="utf-8") == expected_macros,
		"LaTeX result macros match comparison",
	)
	if FINAL_PAPER_MAIN.exists() and RESULT_MACRO_START in FINAL_PAPER_MAIN.read_text(encoding="utf-8"):
		main_text = FINAL_PAPER_MAIN.read_text(encoding="utf-8")
		start = main_text.index(RESULT_MACRO_START) + len(RESULT_MACRO_START)
		end = main_text.index(RESULT_MACRO_END)
		require(
			main_text[start:end].strip() == expected_macros.strip(),
			"LaTeX main generated result macro block matches comparison",
		)


def _reset_managed_output(output_dir: Path) -> None:
	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
		encoding="utf-8",
	)


def _resolve_project_path(value: object, *, label: str) -> Path:
	text = str(value or "").strip()
	if not text:
		raise ValueError(f"Missing path for {label}.")
	path = Path(text).expanduser()
	if not path.is_absolute():
		path = PROJECT_ROOT / path
	return path.resolve()


if __name__ == "__main__":
	main()
