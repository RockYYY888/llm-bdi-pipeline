import json
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.pipeline_logger import PipelineLogger


def test_stage3_success_summary_in_execution_json_is_derived_from_method_library(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )

    method_library = {
        "compound_tasks": [
            {"name": "place_on", "parameters": ["B1", "B2"], "is_primitive": False},
            {"name": "hold_block", "parameters": ["B"], "is_primitive": False},
        ],
        "primitive_tasks": [
            {"name": "pick_up_from_table", "parameters": ["B"], "is_primitive": True},
        ],
        "methods": [
            {"method_name": "m_place_on_stack", "task_name": "place_on"},
            {"method_name": "m_place_on_noop", "task_name": "place_on"},
            {"method_name": "m_hold_block_from_table", "task_name": "hold_block"},
        ],
        "target_literals": [
            {
                "predicate": "on",
                "args": ["a", "b"],
                "is_positive": True,
                "source_symbol": "on_a_b",
            },
        ],
        "target_task_bindings": [
            {"target_literal": "on(a, b)", "task_name": "place_on"},
        ],
    }

    logger.log_stage3_method_synthesis(
        method_library,
        "Success",
        model="deepseek-chat",
        llm_prompt={"system": "SYSTEM", "user": "USER"},
        llm_response='{"ok":true}',
        metadata={
            "used_llm": True,
            "llm_attempted": True,
            "llm_response_time_seconds": 4.321,
            "llm_attempt_durations_seconds": [1.111, 3.21],
        },
    )
    logger.end_pipeline(success=True)

    log_dir = logger.current_log_dir
    assert log_dir is not None
    assert log_dir.name.endswith("_BLOCKS_BW-rand-5")

    execution = json.loads((log_dir / "execution.json").read_text())

    assert execution["stage3_status"] == "success"
    assert execution["stage3_method_library"] == {
        "artifact_path": "htn_method_library.json",
    }
    assert execution["stage3_metadata"]["target_literals"] == ["on(a, b)"]
    assert execution["stage3_metadata"]["target_task_bindings"] == [
        {"target_literal": "on(a, b)", "task_name": "place_on"},
    ]
    assert execution["stage3_metadata"]["target_task_binding_count"] == 1
    assert execution["stage3_metadata"]["compound_tasks"] == 2
    assert execution["stage3_metadata"]["compound_task_names"] == ["place_on", "hold_block"]
    assert execution["stage3_metadata"]["primitive_tasks"] == 1
    assert execution["stage3_metadata"]["primitive_task_names"] == ["pick_up_from_table"]
    assert execution["stage3_metadata"]["methods"] == 3
    assert execution["stage3_metadata"]["method_counts_by_task"] == {
        "place_on": 2,
        "hold_block": 1,
    }
    assert execution["stage3_metadata"]["used_llm"] is True
    assert execution["stage3_metadata"]["llm_attempted"] is True
    assert execution["stage3_metadata"]["llm_response_time_seconds"] == 4.321
    assert execution["stage3_metadata"]["llm_attempt_durations_seconds"] == [1.111, 3.21]


def test_stage3_failure_persists_diagnostics_in_logs(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )

    logger.log_stage3_method_synthesis(
        None,
        "Failed",
        error="HTN synthesis failed during response_parse: invalid payload",
        model="deepseek-chat",
        llm_prompt={"system": "SYSTEM", "user": "USER"},
        llm_response='{"bad":"payload"}',
        metadata={
            "used_llm": True,
            "model": "deepseek-chat",
            "failure_stage": "response_parse",
            "failure_reason": "invalid payload",
            "failure_class": "llm_response_parse_failed",
        },
    )
    logger.end_pipeline(success=False)

    log_dir = logger.current_log_dir
    assert log_dir is not None

    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["stage3_status"] == "failed"
    assert execution["stage3_used_llm"] is True
    assert execution["stage3_model"] == "deepseek-chat"
    assert execution["stage3_llm_prompt"]["system"] == "SYSTEM"
    assert execution["stage3_llm_response"] == '{"bad":"payload"}'
    assert execution["stage3_metadata"]["failure_stage"] == "response_parse"
    assert execution["stage3_metadata"]["failure_reason"] == "invalid payload"
    assert execution["stage3_metadata"]["failure_class"] == "llm_response_parse_failed"

    assert "HTN METHOD SYNTHESIS DIAGNOSTICS" in execution_txt
    assert "failure_stage: response_parse" in execution_txt
    assert "failure_reason: invalid payload" in execution_txt
    assert "LLM RESPONSE (Stage 3)" in execution_txt
    assert "Error: HTN synthesis failed during response_parse: invalid payload" in execution_txt


def test_stage3_summary_uses_bang_signature_for_negative_literals(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )

    method_library = {
        "compound_tasks": [
            {"name": "keep_not_clear", "parameters": ["B"], "is_primitive": False},
        ],
        "primitive_tasks": [],
        "methods": [
            {"method_name": "m_keep_not_clear_noop", "task_name": "keep_not_clear"},
        ],
        "target_literals": [
            {
                "predicate": "clear",
                "args": ["a"],
                "is_positive": False,
                "negation_mode": "naf",
                "source_symbol": "clear_a",
            },
        ],
        "target_task_bindings": [
            {"target_literal": "!clear(a)", "task_name": "keep_not_clear"},
        ],
    }

    logger.log_stage3_method_synthesis(
        method_library,
        "Success",
        metadata={"negation_resolution": {"mode_by_predicate": {"clear/1": "naf"}}},
    )
    logger.end_pipeline(success=True)

    execution = json.loads((logger.current_log_dir / "execution.json").read_text())
    assert execution["stage3_metadata"]["target_literals"] == ["!clear(a)"]
    assert execution["negation_resolution"]["mode_by_predicate"] == {"clear/1": "naf"}


def test_pipeline_logger_records_run_origin_and_logs_root(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path / "tests_logs"), run_origin="tests")
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )
    logger.end_pipeline(success=True)

    log_dir = logger.current_log_dir
    assert log_dir is not None

    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["run_origin"] == "tests"
    assert execution["logs_root"] == str((tmp_path / "tests_logs").resolve())
    assert execution["domain_name"] == "BLOCKS"
    assert execution["problem_name"] == "BW-rand-5"
    assert "Run Origin: tests" in execution_txt
    assert "Domain Name: BLOCKS" in execution_txt
    assert "Problem Name: BW-rand-5" in execution_txt


def test_pipeline_logger_persists_stage7_verification_details(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        problem_file="problem.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )
    logger.log_stage7_official_verification(
        {
            "tool_available": True,
            "plan_kind": "hierarchical",
            "verification_result": True,
            "primitive_plan_executable": True,
            "reached_goal_state": True,
        },
        "Success",
        metadata={
            "backend": "pandaPIparser",
            "status": "success",
            "plan_kind": "hierarchical",
        },
    )
    logger.end_pipeline(success=True)

    log_dir = logger.current_log_dir
    assert log_dir is not None
    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["stage7_status"] == "success"
    assert execution["stage7_backend"] == "pandaPIparser"
    assert execution["stage7_artifacts"]["verification_result"] is True
    assert "STAGE 7: Official IPC HTN Plan Verification" in execution_txt
    assert "OFFICIAL VERIFICATION SUMMARY" in execution_txt


def test_pipeline_logger_persists_structured_timing_profile(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="BLOCKS",
        problem_name="BW-rand-5",
        output_dir=str(tmp_path),
    )
    logger.record_stage_timing(
        "stage1",
        1.234567,
        breakdown={
            "prompt_build_seconds": 0.111111,
            "llm_roundtrip_seconds": 1.0,
        },
        metadata={"prefer_skeletal_task_grounded_output": True},
    )
    logger.end_pipeline(success=True)

    log_dir = logger.current_log_dir
    assert log_dir is not None
    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["timing_profile"]["stage1"]["duration_seconds"] == 1.234567
    assert execution["timing_profile"]["stage1"]["breakdown_seconds"] == {
        "prompt_build_seconds": 0.111111,
        "llm_roundtrip_seconds": 1.0,
    }
    assert execution["timing_profile"]["stage1"]["metadata"] == {
        "prefer_skeletal_task_grounded_output": True,
    }
    assert "TIMING PROFILE" in execution_txt
    assert "stage1: 1.235s" in execution_txt
    assert "prompt_build_seconds: 0.111s" in execution_txt


def test_pipeline_logger_sanitises_stage6_artifact_path_lists(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        domain_name="ROVER",
        problem_name="roverprob1234",
        output_dir=str(tmp_path),
    )
    log_dir = logger.current_log_dir
    assert log_dir is not None

    chunk_a = log_dir / "chunk_a.plan"
    chunk_b = log_dir / "chunk_b.plan"
    chunk_a.write_text("a")
    chunk_b.write_text("b")

    logger.log_stage6_jason_validation(
        {
            "backend": "RunLocalMAS",
            "status": "failed",
            "artifacts": {
                "guided_chunk_plans": [str(chunk_a), str(chunk_b)],
            },
        },
        "Failed",
        error="timeout",
        metadata={"backend": "RunLocalMAS"},
    )
    logger.end_pipeline(success=False)

    execution = json.loads((log_dir / "execution.json").read_text())
    assert execution["stage6_status"] == "failed"
    assert execution["stage6_artifacts"]["artifacts"]["guided_chunk_plans"] == [
        "chunk_a.plan",
        "chunk_b.plan",
    ]
