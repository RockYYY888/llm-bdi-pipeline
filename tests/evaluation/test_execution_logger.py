from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from execution_logging.execution_logger import ExecutionLogger


def test_execution_logger_writes_only_active_semantic_steps(tmp_path) -> None:
	logger = ExecutionLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"stack block c on block b",
		mode="plan_library_evaluation",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		domain_name="blocksworld",
		problem_name="p01",
	)
	logger.log_goal_grounding_success(
		{
			"ltlf_formula": "F(subgoal_1)",
			"subgoals": [{"id": "subgoal_1", "task_name": "stack", "args": ["c", "b"]}],
		},
		used_llm=True,
		model="test-goal-grounding-model",
		llm_prompt={"system": "sys", "user": "usr"},
		llm_response='{"ltlf_formula":"F(subgoal_1)","subgoals":[{"id":"subgoal_1","task_name":"stack","args":["c","b"]}]}',
		metadata={
			"evaluation_domain_source": "benchmark",
			"llm_response_mode": "streaming",
			"llm_finish_reason": "stop",
		},
	)
	logger.log_agentspeak_rendering(
		{"asl_file": "/tmp/query_runtime.asl"},
		"Success",
		metadata={"task_event_count": 1},
	)
	logger.log_runtime_execution(
		{"action_path": ["pickup(c)", "stack(c, b)"]},
		"Success",
		backend="RunLocalMAS",
		metadata={"step_count": 2},
	)
	logger.record_failure_signature(
		{
			"ltlf_formula": "F(stack(c, b))",
			"ltlf_atom_count": 1,
			"ltlf_operator_counts": {"F": 1},
			"jason_failure_class": None,
			"failed_goals": [],
			"verifier_missing_goal_facts": [],
		},
	)
	logger.log_plan_solve({"step_count": 2}, "Success", metadata={"backend": "jason"})
	logger.log_official_verification(
		{"verification_result": True},
		"Success",
		metadata={"backend": "pandaPIparser"},
	)
	log_path = logger.end_pipeline(success=True)
	log_dir = log_path.parent

	execution = json.loads((log_dir / "execution.json").read_text())
	text_log = (log_dir / "execution.txt").read_text()

	assert "goal_grounding" in execution
	assert "agentspeak_rendering" in execution
	assert "runtime_execution" in execution
	assert "plan_solve" in execution
	assert "plan_verification" in execution
	assert "method_synthesis" not in execution
	assert "domain_gate" not in execution
	assert "stage1_status" not in execution
	assert "pending" not in text_log
	assert "PLAN LIBRARY EVALUATION" in text_log
	assert "GOAL GROUNDING" in text_log
	assert "AGENTSPEAK RENDERING" in text_log
	assert "RUNTIME EXECUTION" in text_log
	assert "PLAN SOLVE" in text_log
	assert "OFFICIAL VERIFICATION" in text_log
	assert "STAGE 1" not in text_log
	assert '"ltlf_formula": "F(subgoal_1)"' in text_log
	assert '"llm_response_mode": "streaming"' in text_log
	assert '"llm_finish_reason": "stop"' in text_log
	assert execution["goal_grounding"]["metadata"]["llm_response_mode"] == "streaming"
	assert execution["goal_grounding"]["metadata"]["llm_finish_reason"] == "stop"
	assert "prompt" not in execution["goal_grounding"]["llm"]
	assert "response" not in execution["goal_grounding"]["llm"]
	assert execution["goal_grounding"]["llm"]["prompt_file"].endswith(
		"goal_grounding_llm_prompt.json",
	)
	assert execution["goal_grounding"]["llm"]["response_file"].endswith(
		"goal_grounding_llm_response.txt",
	)
	assert (log_dir / execution["goal_grounding"]["llm"]["prompt_file"]).exists()
	assert (log_dir / execution["goal_grounding"]["llm"]["response_file"]).exists()
	assert execution["ltlf_formula"] == "F(stack(c, b))"
	assert execution["ltlf_atom_count"] == 1
	assert execution["ltlf_operator_counts"] == {"F": 1}


def test_execution_logger_compacts_long_query_and_formula_text(tmp_path) -> None:
	logger = ExecutionLogger(logs_dir=str(tmp_path), run_origin="tests")
	long_query = "move block " * 600
	long_formula = " & ".join(f"do_move(b{i},b{i + 1})" for i in range(500))
	logger.start_pipeline(
		long_query,
		mode="plan_library_evaluation",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		domain_name="blocksworld",
		problem_name="p30",
	)
	logger.record_failure_signature(
		{
			"ltlf_formula": long_formula,
			"ltlf_atom_count": 500,
			"ltlf_operator_counts": {"&": 499},
		},
	)
	log_path = logger.end_pipeline(success=True)

	execution = json.loads((log_path.parent / "execution.json").read_text())

	assert len(execution["natural_language"]) < len(long_query)
	assert "truncated" in execution["natural_language"]
	assert len(execution["ltlf_formula"]) < len(long_formula)
	assert execution["failure_signature"]["ltlf_formula_truncated"] is True
	assert execution["failure_signature"]["ltlf_formula_chars"] == len(long_formula)
	assert len(execution["failure_signature"]["ltlf_formula_sha256"]) == 64
	assert execution["ltlf_atom_count"] == 500


def test_execution_logger_serializes_backend_bytes_payloads(tmp_path) -> None:
	logger = ExecutionLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"official root task",
		mode="official_problem_root_execution",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		domain_name="blocksworld",
		problem_name="p30",
	)
	logger.log_plan_solve(
		{
			"planner_attempts": [
				{
					"planner_id": "lifted_panda_sat",
					"stdout": b"solver-bytes",
					"stderr": b"",
				},
			],
		},
		"Success",
	)
	log_path = logger.end_pipeline(success=True)
	execution = json.loads((log_path.parent / "execution.json").read_text())
	text_log = log_path.read_text()

	assert execution["plan_solve"]["artifacts"]["planner_attempts"][0]["stdout"] == "solver-bytes"
	assert "OFFICIAL PROBLEM ROOT EXECUTION" in text_log


def test_execution_logger_externalizes_method_and_plan_payloads(tmp_path) -> None:
	logger = ExecutionLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"generate library",
		mode="plan_library_generation",
		domain_file="/tmp/domain.hddl",
		domain_name="blocksworld",
	)
	logger.log_method_synthesis(
		{
			"methods": [
				{"name": "m1", "subtasks": [{"id": "s1", "task_name": "pick-up"}]},
			],
		},
		"Success",
		model="deepseek-v4-pro",
		llm_prompt={"system": "system prompt", "user": "user prompt"},
		llm_response='{"methods":[]}',
	)
	logger.log_agentspeak_rendering(
		{"plan_library": {"plans": [{"trigger": "+!do_put_on"}]}},
		"Success",
	)
	log_path = logger.end_pipeline(success=True)
	log_dir = log_path.parent
	execution = json.loads((log_dir / "execution.json").read_text())
	text_log = log_path.read_text()

	method_artifacts = execution["method_synthesis"]["artifacts"]
	render_artifacts = execution["agentspeak_rendering"]["artifacts"]

	assert "method_library" not in method_artifacts
	assert method_artifacts["method_library_file"].endswith(
		"payloads/method_synthesis_method_library.json",
	)
	assert render_artifacts["plan_library_file"].endswith(
		"payloads/agentspeak_rendering_plan_library.json",
	)
	assert '{"methods":[]}' not in text_log
	assert (log_dir / method_artifacts["method_library_file"]).exists()
	assert (log_dir / render_artifacts["plan_library_file"]).exists()
