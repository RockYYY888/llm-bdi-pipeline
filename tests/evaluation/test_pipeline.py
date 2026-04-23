from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from evaluation import PlanLibraryEvaluationPipeline
from method_library import HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from evaluation.orchestrator import PlanLibraryEvaluationOrchestrator
from plan_library import (
	LibraryValidationRecord,
	PlanLibraryArtifactBundle,
	build_plan_library,
)
from temporal_specification import QueryInstructionRecord, TemporalSpecificationRecord
from tests.support.plan_library_generation_support import DOMAIN_FILES
from utils.hddl_parser import HDDLParser


def _sample_bundle() -> PlanLibraryArtifactBundle:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(name="do_put_on", parameters=("?x", "?y"), is_primitive=False),
		],
		primitive_tasks=[
			HTNTask(name="pick_up", parameters=("?x",), is_primitive=True),
			HTNTask(name="stack", parameters=("?x", "?y"), is_primitive=True),
		],
		methods=[
			HTNMethod(
				method_name="m_do_put_on_serial",
				task_name="do_put_on",
				parameters=("?x", "?y"),
				task_args=("?x", "?y"),
				subtasks=(
					HTNMethodStep("s1", "pick_up", ("?x",), "primitive", action_name="pick_up"),
					HTNMethodStep("s2", "stack", ("?x", "?y"), "primitive", action_name="stack"),
				),
				ordering=(("s1", "s2"),),
				source_instruction_ids=("query_1",),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)
	plan_library, translation_coverage = build_plan_library(
		domain=domain,
		method_library=method_library,
	)
	return PlanLibraryArtifactBundle(
		domain_name=domain.name,
		query_sequence=(
			QueryInstructionRecord(
				instruction_id="query_1",
				source_text="Put block b4 on block b2.",
				problem_file="p01.hddl",
			),
		),
		temporal_specifications=(
			TemporalSpecificationRecord(
				instruction_id="query_1",
				source_text="Put block b4 on block b2.",
				ltlf_formula="do_put_on(b4,b2)",
				referenced_events=(),
				diagnostics=(),
				problem_file="p01.hddl",
			),
		),
		method_library=method_library,
		plan_library=plan_library,
		translation_coverage=translation_coverage,
		library_validation=LibraryValidationRecord(
			library_id=domain.name,
			passed=True,
			method_count=1,
			plan_count=1,
			checked_layers={
				"signature_conformance": True,
				"typed_structure": True,
				"body_symbol_validity": True,
				"groundability_precheck": True,
			},
		),
		method_synthesis_metadata={"model": "test-model"},
		artifact_root=str(PROJECT_ROOT / "artifacts" / "plan_library" / "blocksworld"),
	)


def test_benchmark_evaluation_uses_stored_temporal_specification(
	tmp_path: Path,
	monkeypatch,
) -> None:
	captured: dict[str, object] = {}

	def fake_execute_grounded_query_with_library(
		self,
		nl_query,
		*,
		library_artifact,
		grounding_result,
		execution_mode="plan_library_evaluation",
	):
		_ = library_artifact
		captured["nl_query"] = nl_query
		captured["grounding_result"] = grounding_result
		captured["execution_mode"] = execution_mode
		log_path = tmp_path / "runs" / "benchmark_case" / "execution.txt"
		log_path.parent.mkdir(parents=True, exist_ok=True)
		log_path.write_text("", encoding="utf-8")
		return {"success": True, "log_path": str(log_path)}

	monkeypatch.setattr(
		PlanLibraryEvaluationOrchestrator,
		"execute_grounded_query_with_library",
		fake_execute_grounded_query_with_library,
	)

	pipeline = PlanLibraryEvaluationPipeline(domain_file=DOMAIN_FILES["blocksworld"])
	result = pipeline.evaluate_benchmark_case(
		library_artifact=_sample_bundle(),
		query_id="query_1",
	)

	assert result["success"] is True
	assert captured["execution_mode"] == "plan_library_evaluation"
	grounding_result = captured["grounding_result"]
	assert grounding_result.ltlf_formula == "do_put_on(b4, b2) & X(do_put_on(b1, b4) & X(do_put_on(b3, b1)))"
	assert grounding_result.subgoals[0].task_name == "do_put_on"
	report = json.loads(Path(result["evaluation_report_path"]).read_text())
	assert report["evaluation_mode"] == "stored_benchmark_case"
	assert report["query_id"] == "query_1"


def test_ad_hoc_evaluation_with_explicit_formula_skips_live_grounding(
	tmp_path: Path,
	monkeypatch,
) -> None:
	captured: dict[str, object] = {}

	def fake_execute_grounded_query_with_library(
		self,
		nl_query,
		*,
		library_artifact,
		grounding_result,
		execution_mode="plan_library_evaluation",
	):
		_ = library_artifact
		captured["nl_query"] = nl_query
		captured["grounding_result"] = grounding_result
		captured["execution_mode"] = execution_mode
		log_path = tmp_path / "runs" / "ad_hoc_case" / "execution.txt"
		log_path.parent.mkdir(parents=True, exist_ok=True)
		log_path.write_text("", encoding="utf-8")
		return {"success": True, "log_path": str(log_path)}

	monkeypatch.setattr(
		PlanLibraryEvaluationOrchestrator,
		"execute_grounded_query_with_library",
		fake_execute_grounded_query_with_library,
	)

	problem_file = Path(DOMAIN_FILES["blocksworld"]).resolve().parent / "problems" / "p01.hddl"
	pipeline = PlanLibraryEvaluationPipeline(domain_file=DOMAIN_FILES["blocksworld"])
	result = pipeline.evaluate_instruction(
		library_artifact=_sample_bundle(),
		instruction="Put block b4 on block b2.",
		problem_file=str(problem_file),
		ltlf_formula="do_put_on(b4,b2)",
	)

	assert result["success"] is True
	assert captured["execution_mode"] == "plan_library_evaluation"
	grounding_result = captured["grounding_result"]
	assert grounding_result.ltlf_formula == "do_put_on(b4,b2)"
	assert grounding_result.subgoals[0].task_name == "do_put_on"
	report = json.loads(Path(result["evaluation_report_path"]).read_text())
	assert report["evaluation_mode"] == "ad_hoc_temporal_specification"


def test_evaluation_report_falls_back_to_tmp_root_without_log_path(
	tmp_path: Path,
	monkeypatch,
) -> None:
	def fake_execute_query_with_library(
		self,
		nl_query,
		*,
		library_artifact,
		execution_mode="plan_library_evaluation",
	):
		_ = self
		_ = nl_query
		_ = library_artifact
		_ = execution_mode
		return {"success": True}

	monkeypatch.setattr(
		PlanLibraryEvaluationOrchestrator,
		"execute_query_with_library",
		fake_execute_query_with_library,
	)

	problem_file = Path(DOMAIN_FILES["blocksworld"]).resolve().parent / "problems" / "p01.hddl"
	bundle = _sample_bundle()
	pipeline = PlanLibraryEvaluationPipeline(domain_file=DOMAIN_FILES["blocksworld"])
	result = pipeline.evaluate_instruction(
		library_artifact=bundle,
		instruction="Put block b4 on block b2.",
		problem_file=str(problem_file),
	)

	report_path = Path(result["evaluation_report_path"]).resolve()
	assert report_path == (
		PROJECT_ROOT
		/ "tmp"
		/ "evaluation"
		/ str(bundle.domain_name)
		/ "evaluation_report.json"
	).resolve()
	assert json.loads(report_path.read_text())["evaluation_mode"] == "ad_hoc_live_grounding"
