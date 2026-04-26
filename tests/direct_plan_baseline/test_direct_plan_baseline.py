from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from evaluation.direct_plan_baseline import (
	DirectPlanParseError,
	build_direct_plan_system_prompt,
	build_direct_plan_user_prompt,
	materialize_plan_text,
	parse_direct_plan_response,
	run_direct_plan_baseline_case,
)
from temporal_specification.models import TemporalSpecificationRecord
from utils.hddl_parser import HDDLParser


def test_direct_plan_prompt_is_case_level_not_library_level() -> None:
	domain = HDDLParser.parse_domain(
		str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl"),
	)
	problem = HDDLParser.parse_problem(
		str(PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl"),
	)
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="complete the tasks",
		ltlf_formula="do_put_on(b4, b2) & X(do_put_on(b1, b4))",
		referenced_events=(),
		problem_file="p01.hddl",
	)

	system_prompt = build_direct_plan_system_prompt()
	user_prompt = build_direct_plan_user_prompt(
		domain=domain,
		problem=problem,
		temporal_specification=temporal_specification,
		instruction="complete the tasks",
	)

	assert "verifier-readable primitive plan" in system_prompt
	assert "HTN methods" in system_prompt
	assert "AgentSpeak" not in user_prompt
	assert "objects:" in user_prompt
	assert "initial_state:" in user_prompt
	assert "goal_condition:" in user_prompt
	assert "do_put_on(b4, b2)" in user_prompt
	assert "0 pick-up b1" in user_prompt


def test_parse_direct_plan_response_accepts_strict_json() -> None:
	response = json.dumps(
		{
			"plan_lines": ["0 pick-up b1", "1 stack b1 b2"],
			"diagnostics": ["uses declared actions"],
		},
	)

	parsed = parse_direct_plan_response(response)

	assert parsed["plan_lines"] == ["0 pick-up b1", "1 stack b1 b2"]
	assert parsed["diagnostics"] == ["uses declared actions"]


def test_parse_direct_plan_response_rejects_unknown_keys() -> None:
	with pytest.raises(DirectPlanParseError, match="unexpected keys"):
		parse_direct_plan_response(
			json.dumps(
				{
					"plan_lines": [],
					"diagnostics": [],
					"proof": "not allowed",
				},
			),
		)


def test_materialize_plan_text_adds_verifier_wrapper_once() -> None:
	plan_text = materialize_plan_text(["==>", "0 pick-up b1", "1 stack b1 b2", "root"])

	assert plan_text == "==>\n0 pick-up b1\n1 stack b1 b2\nroot\n"


def test_run_direct_plan_case_can_persist_without_verifier(tmp_path: Path) -> None:
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="complete the tasks",
		ltlf_formula="do_put_on(b4, b2)",
		referenced_events=(),
		problem_file="p01.hddl",
	)
	response_text = json.dumps(
		{
			"plan_lines": ["0 unstack b2 b3", "1 put-down b2"],
			"diagnostics": ["collection-only smoke"],
		},
	)

	result = run_direct_plan_baseline_case(
		domain_key="blocksworld",
		query_id="query_1",
		domain_file=PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl",
		problem_file=PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl",
		instruction="complete the tasks",
		temporal_specification=temporal_specification,
		output_dir=tmp_path,
		response_text=response_text,
		verify=False,
	)

	validation = json.loads((tmp_path / "direct_plan_validation.json").read_text())
	assert result.parseable is True
	assert result.verification_skipped is True
	assert validation["verification_skipped"] is True
	assert (tmp_path / "plan.txt").read_text() == "==>\n0 unstack b2 b3\n1 put-down b2\nroot\n"


def test_run_direct_plan_case_marks_parse_failure_as_collection_only(tmp_path: Path) -> None:
	temporal_specification = TemporalSpecificationRecord(
		instruction_id="query_1",
		source_text="complete the tasks",
		ltlf_formula="do_put_on(b4, b2)",
		referenced_events=(),
		problem_file="p01.hddl",
	)

	result = run_direct_plan_baseline_case(
		domain_key="blocksworld",
		query_id="query_1",
		domain_file=PROJECT_ROOT / "src" / "domains" / "blocksworld" / "domain.hddl",
		problem_file=PROJECT_ROOT / "src" / "domains" / "blocksworld" / "problems" / "p01.hddl",
		instruction="complete the tasks",
		temporal_specification=temporal_specification,
		output_dir=tmp_path,
		response_text="not json",
		verify=False,
	)

	validation = json.loads((tmp_path / "direct_plan_validation.json").read_text())
	assert result.parseable is False
	assert result.verification_skipped is True
	assert validation["verification_skipped"] is True
