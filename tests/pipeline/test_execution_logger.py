from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from pipeline.execution_logger import ExecutionLogger


def test_execution_logger_writes_only_active_semantic_steps(tmp_path) -> None:
	logger = ExecutionLogger(logs_dir=str(tmp_path), run_origin="tests")
	logger.start_pipeline(
		"stack block c on block b",
		mode="query_execution",
		domain_file="/tmp/domain.hddl",
		problem_file="/tmp/problem.hddl",
		domain_name="blocksworld",
		problem_name="p01",
	)
	logger.log_goal_grounding_success(
		{"task_network": [{"task_name": "stack", "args": ["c", "b"]}]},
		used_llm=False,
		model=None,
		llm_prompt=None,
		llm_response=None,
	)
	logger.log_plan_solve({"step_count": 2}, "Success", metadata={"backend": "pandaPI"})
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
	assert "plan_solve" in execution
	assert "plan_verification" in execution
	assert "method_synthesis" not in execution
	assert "domain_gate" not in execution
	assert "stage1_status" not in execution
	assert "pending" not in text_log
	assert "GOAL GROUNDING" in text_log
	assert "PLAN SOLVE" in text_log
	assert "OFFICIAL VERIFICATION" in text_log
	assert "STAGE 1" not in text_log


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
			"backend_attempts": [
				{
					"backend_name": "pandadealer_agile_lama",
					"stdout": b"solver-bytes",
					"stderr": b"",
				},
			],
		},
		"Success",
	)
	log_path = logger.end_pipeline(success=True)
	execution = json.loads((log_path.parent / "execution.json").read_text())

	assert execution["plan_solve"]["artifacts"]["backend_attempts"][0]["stdout"] == "solver-bytes"
