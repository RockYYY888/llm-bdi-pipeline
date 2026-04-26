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

from method_library import HTNMethod, HTNMethodLibrary, HTNMethodStep, HTNTask
from plan_library import PlanLibraryGenerationPipeline
from tests.support.plan_library_generation_support import DOMAIN_FILES


def _sample_blocksworld_method_library() -> HTNMethodLibrary:
	return HTNMethodLibrary(
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


def test_plan_library_generation_pipeline_persists_chapter4_artifacts(
	tmp_path: Path,
) -> None:
	domain_file = DOMAIN_FILES["blocksworld"]
	pipeline = PlanLibraryGenerationPipeline(domain_file=domain_file)
	masked_domain_text = Path(domain_file).read_text(encoding="utf-8")
	masked_domain_file = tmp_path / "masked_domain.hddl"
	masked_domain_file.write_text(masked_domain_text, encoding="utf-8")

	pipeline._orchestrator.prepare_masked_domain_build_inputs = lambda: {  # type: ignore[method-assign]
		"masked_domain": pipeline.context.domain,
		"masked_domain_file": masked_domain_file,
		"masked_domain_text": masked_domain_text,
		"original_method_count": 3,
	}
	pipeline._orchestrator.synthesise_domain_methods = lambda **_kwargs: (  # type: ignore[method-assign]
		_sample_blocksworld_method_library(),
		{"model": "test-model", "llm_attempted": False},
	)
	pipeline._orchestrator.validate_method_library = lambda _library, **_kwargs: {  # type: ignore[method-assign]
		"layers": {
			"signature_conformance": {"passed": True, "warnings": []},
			"typed_structural_soundness": {"passed": True, "warnings": []},
			"decomposition_admissibility": {"passed": True, "warnings": []},
			"materialized_parseability": {"passed": True, "warnings": []},
		},
	}

	result = pipeline.build_library_bundle(output_root=str(tmp_path / "artifact_bundle"))

	assert result["success"] is True
	artifact_paths = result["artifact_paths"]
	assert Path(artifact_paths["masked_domain"]).exists()
	assert Path(artifact_paths["generated_domain"]).exists()
	assert Path(artifact_paths["query_sequence"]).exists()
	assert Path(artifact_paths["temporal_specifications"]).exists()
	assert Path(artifact_paths["method_library"]).exists()
	assert Path(artifact_paths["plan_library"]).exists()
	assert Path(artifact_paths["translation_coverage"]).exists()
	assert Path(artifact_paths["library_validation"]).exists()
	assert Path(artifact_paths["plan_library_asl"]).exists()

	query_sequence = json.loads(Path(artifact_paths["query_sequence"]).read_text())
	temporal_specifications = json.loads(Path(artifact_paths["temporal_specifications"]).read_text())
	plan_library = json.loads(Path(artifact_paths["plan_library"]).read_text())
	translation_coverage = json.loads(Path(artifact_paths["translation_coverage"]).read_text())
	library_validation = json.loads(Path(artifact_paths["library_validation"]).read_text())

	assert query_sequence[0]["instruction_id"] == "query_1"
	assert temporal_specifications[0]["instruction_id"] == "query_1"
	assert plan_library["plans"][0]["source_instruction_ids"] == ["query_1"]
	assert translation_coverage["accepted_translation"] == 1
	assert library_validation["passed"] is True


def test_plan_library_generation_pipeline_filters_selected_query_ids(
	tmp_path: Path,
) -> None:
	domain_file = DOMAIN_FILES["blocksworld"]
	pipeline = PlanLibraryGenerationPipeline(
		domain_file=domain_file,
		query_ids=("query_3", "query_1"),
	)
	masked_domain_text = Path(domain_file).read_text(encoding="utf-8")
	masked_domain_file = tmp_path / "masked_domain.hddl"
	masked_domain_file.write_text(masked_domain_text, encoding="utf-8")
	captured: dict[str, object] = {}

	pipeline._orchestrator.prepare_masked_domain_build_inputs = lambda: {  # type: ignore[method-assign]
		"masked_domain": pipeline.context.domain,
		"masked_domain_file": masked_domain_file,
		"masked_domain_text": masked_domain_text,
		"original_method_count": 3,
	}

	def fake_synthesise_domain_methods(**kwargs):
		captured["query_sequence"] = kwargs["query_sequence"]
		captured["temporal_specifications"] = kwargs["temporal_specifications"]
		return (
			_sample_blocksworld_method_library(),
			{"model": "test-model", "llm_attempted": False},
		)

	pipeline._orchestrator.synthesise_domain_methods = fake_synthesise_domain_methods  # type: ignore[method-assign]
	pipeline._orchestrator.validate_method_library = lambda _library, **_kwargs: {  # type: ignore[method-assign]
		"layers": {
			"signature_conformance": {"passed": True, "warnings": []},
			"typed_structural_soundness": {"passed": True, "warnings": []},
			"decomposition_admissibility": {"passed": True, "warnings": []},
			"materialized_parseability": {"passed": True, "warnings": []},
		},
	}

	result = pipeline.build_library_bundle(output_root=str(tmp_path / "artifact_bundle"))

	assert result["success"] is True
	assert [
		record["instruction_id"]
		for record in captured["query_sequence"]
	] == ["query_3", "query_1"]
	assert [
		record["instruction_id"]
		for record in captured["temporal_specifications"]
	] == ["query_3", "query_1"]
	query_sequence = json.loads(Path(result["artifact_paths"]["query_sequence"]).read_text())
	assert [record["instruction_id"] for record in query_sequence] == ["query_3", "query_1"]


def test_plan_library_generation_pipeline_scopes_default_artifact_root_by_query_selection() -> None:
	pipeline = PlanLibraryGenerationPipeline(
		domain_file=DOMAIN_FILES["blocksworld"],
		query_ids=("query_1",),
	)

	assert pipeline._default_artifact_root("blocksworld") == (
		PROJECT_ROOT / "artifacts" / "plan_library" / "blocksworld" / "query_1"
	)
