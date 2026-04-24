"""
Tests for mainline natural-language to LTLf dataset generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.artifacts import TemporalGroundingResult
from evaluation.goal_grounding.grounder import NLToLTLfGenerator
from temporal_specification.ltlf_dataset_generation import generate_ltlf_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCKSWORLD_DOMAIN_FILE = PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"


class FakeConfig:
	ltlf_generation_api_key = "sk-test"
	ltlf_generation_model = "deepseek-v4-pro"
	ltlf_generation_base_url = "https://api.deepseek.com"
	ltlf_generation_timeout = 10
	ltlf_generation_max_tokens = 256
	ltlf_generation_session_id = "ltlf-test-session"


class CapturingGenerator:
	calls: list[dict[str, object]] = []

	def __init__(self, **kwargs) -> None:
		self.kwargs = dict(kwargs)

	def generate(
		self,
		nl_instruction: str,
		*,
		method_library,
		typed_objects,
		task_type_map,
		type_parent_map,
	):
		self.calls.append(
			{
				"nl_instruction": nl_instruction,
				"method_library": method_library,
				"typed_objects": dict(typed_objects),
				"task_type_map": dict(task_type_map),
				"type_parent_map": dict(type_parent_map),
				"kwargs": dict(self.kwargs),
			},
		)
		return (
			TemporalGroundingResult(
				query_text=nl_instruction,
				ltlf_formula="do_put_on(b4, b2)",
				subgoals=(),
				typed_objects=dict(typed_objects),
				query_object_inventory=(),
				diagnostics=(),
			),
			{"system": "prompt", "user": "query"},
			'{"ltlf_formula":"do_put_on(b4, b2)"}',
		)


def test_generate_ltlf_dataset_from_natural_language_queries(tmp_path: Path) -> None:
	source_dataset = tmp_path / "benchmark_queries.json"
	output_dataset = tmp_path / "queries_LTLf.json"
	source_dataset.write_text(
		json.dumps(
			{
				"version": 5,
				"dataset_kind": "stored_benchmark_queries",
				"query_protocol_document": "docs/query_protocol.md",
				"domains": {
					"blocksworld": {
						"cases": {
							"query_1": {
								"instruction": (
									"Using blocks b4 and b2, complete the tasks "
									"do_put_on(b4, b2)."
								),
								"problem_file": "p01.hddl",
							},
						},
					},
				},
			},
		),
		encoding="utf-8",
	)
	CapturingGenerator.calls = []

	result = generate_ltlf_dataset(
		source_query_dataset=source_dataset,
		output_dataset=output_dataset,
		domain_file=BLOCKSWORLD_DOMAIN_FILE,
		query_domain="blocksworld",
		query_ids=("query_1",),
		config=FakeConfig(),
		generator_factory=CapturingGenerator,
	)

	assert result["success"] is True
	assert result["total_generated"] == 1
	assert result["total_reused"] == 0
	assert output_dataset.exists()

	payload = json.loads(output_dataset.read_text(encoding="utf-8"))
	case_payload = payload["domains"]["blocksworld"]["cases"]["query_1"]
	assert payload["dataset_kind"] == "stored_benchmark_ltlf_queries"
	assert payload["ltlf_generator"]["base_url"] == "https://api.deepseek.com"
	assert payload["ltlf_generator"]["session_id"] == "ltlf-test-session"
	assert case_payload["ltlf_formula"] == "do_put_on(b4, b2)"
	assert case_payload["instruction"].startswith("Using blocks")

	assert len(CapturingGenerator.calls) == 1
	call = CapturingGenerator.calls[0]
	assert call["method_library"] is None
	assert call["typed_objects"]["b4"] == "block"
	assert call["task_type_map"]["do_put_on"] == ("block", "block")
	assert call["type_parent_map"]["block"] == "object"
	assert call["kwargs"]["session_id"] == "ltlf-test-session"


def test_generate_ltlf_dataset_reuses_existing_formula_without_generator(tmp_path: Path) -> None:
	source_dataset = tmp_path / "queries_LTLf.json"
	output_dataset = tmp_path / "output_queries_LTLf.json"
	source_dataset.write_text(
		json.dumps(
			{
				"domains": {
					"blocksworld": {
						"cases": {
							"query_1": {
								"instruction": "Using blocks b4 and b2, complete the tasks do_put_on(b4, b2).",
								"problem_file": "p01.hddl",
								"ltlf_formula": "F(do_put_on(b4, b2))",
							},
						},
					},
				},
			},
		),
		encoding="utf-8",
	)
	CapturingGenerator.calls = []

	result = generate_ltlf_dataset(
		source_query_dataset=source_dataset,
		output_dataset=output_dataset,
		domain_file=BLOCKSWORLD_DOMAIN_FILE,
		query_domain="blocksworld",
		query_ids=("query_1",),
		config=FakeConfig(),
		generator_factory=CapturingGenerator,
	)

	assert result["total_generated"] == 0
	assert result["total_reused"] == 1
	assert CapturingGenerator.calls == []
	payload = json.loads(output_dataset.read_text(encoding="utf-8"))
	assert payload["domains"]["blocksworld"]["cases"]["query_1"]["ltlf_formula"] == (
		"F(do_put_on(b4, b2))"
	)


def test_goal_grounding_prompt_lists_domain_tasks_without_method_library() -> None:
	generator = NLToLTLfGenerator(domain_file=str(BLOCKSWORLD_DOMAIN_FILE))

	system_prompt, _ = generator._build_prompts(
		query_text="Using blocks b4 and b2, complete the tasks do_put_on(b4, b2).",
		method_library=None,
		typed_objects={"b4": "block", "b2": "block"},
		task_type_map={"do_put_on": ("block", "block")},
	)

	assert "Callable grounded evaluation tasks:" in system_prompt
	assert "- do_put_on(block, block)" in system_prompt
	assert "Callable grounded evaluation tasks:\n- none" not in system_prompt
