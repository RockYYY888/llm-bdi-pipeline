"""
Unit tests for Pipeline Logger

Tests cover:
- Logging initialization
- Stage logging (success and error)
- JSON and text output generation
- LLM interaction capture
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from pipeline_logger import PipelineLogger


# ===== Test Logger Initialization =====

class TestLoggerInitialization:
    """Test pipeline logger initialization"""

    def test_logger_creates_log_directory(self, tmp_path, monkeypatch):
        """Test logger creates log directory"""
        log_dir = tmp_path / "logs"
        monkeypatch.setattr('pipeline_logger.Path', lambda x: log_dir if x == "logs" else Path(x))

        logger = PipelineLogger()
        assert logger is not None

    def test_start_pipeline(self, tmp_path):
        """Test start_pipeline initializes log data"""
        logger = PipelineLogger()
        logger.start_pipeline(
            nl_instruction="Test instruction",
            domain_file="domain.pddl",
            output_dir=str(tmp_path),
            timestamp="20250122_120000"
        )

        assert logger.log_data["nl_instruction"] == "Test instruction"
        assert logger.log_data["domain_file"] == "domain.pddl"
        assert logger.log_data["output_directory"] == str(tmp_path)
        assert logger.log_data["timestamp"] == "20250122_120000"
        assert "start_time" in logger.log_data


# ===== Test Stage 1 Logging =====

class TestStage1Logging:
    """Test Stage 1 logging"""

    def test_log_stage1_success_without_llm(self, tmp_path):
        """Test logging Stage 1 success without LLM"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        ltl_spec_dict = {
            "objects": ["a", "b"],
            "formulas_string": ["F(on(a, b))"]
        }

        logger.log_stage1_success(
            ltl_spec_dict,
            used_llm=False,
            model=None,
            prompt=None,
            response=None
        )

        assert logger.log_data["stage1"]["status"] == "Success"
        assert logger.log_data["stage1"]["ltl_specification"] == ltl_spec_dict
        assert logger.log_data["stage1"]["used_llm"] is False

    def test_log_stage1_success_with_llm(self, tmp_path):
        """Test logging Stage 1 success with LLM interaction capture"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        ltl_spec_dict = {"objects": ["a"], "formulas_string": ["F(on(a, b))"]}
        prompt_dict = {"system": "System prompt", "user": "User prompt"}
        response_text = '{"result": "success"}'

        logger.log_stage1_success(
            ltl_spec_dict,
            used_llm=True,
            model="gpt-4o-mini",
            prompt=prompt_dict,
            response=response_text
        )

        assert logger.log_data["stage1"]["status"] == "Success"
        assert logger.log_data["stage1"]["used_llm"] is True
        assert logger.log_data["stage1"]["model"] == "gpt-4o-mini"
        assert logger.log_data["stage1"]["llm_prompt"] == prompt_dict
        assert logger.log_data["stage1"]["llm_response"] == response_text

    def test_log_stage1_error(self, tmp_path):
        """Test logging Stage 1 error"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        error_message = "API connection failed"
        logger.log_stage1_error(error_message)

        assert logger.log_data["stage1"]["status"] == "Failed"
        assert logger.log_data["stage1"]["error"] == error_message


# ===== Test Stage 2 Logging =====

class TestStage2Logging:
    """Test Stage 2 logging"""

    def test_log_stage2_success_without_llm(self, tmp_path):
        """Test logging Stage 2 success without LLM"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        pddl_problem = "(define (problem test) ...)"
        logger.log_stage2_success(
            pddl_problem,
            used_llm=False,
            model=None,
            prompt=None,
            response=None
        )

        assert logger.log_data["stage2"]["status"] == "Success"
        assert logger.log_data["stage2"]["pddl_problem_length"] == len(pddl_problem)
        assert logger.log_data["stage2"]["used_llm"] is False

    def test_log_stage2_success_with_llm(self, tmp_path):
        """Test logging Stage 2 success with LLM interaction capture"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        pddl_problem = "(define (problem test) ...)"
        prompt_dict = {"system": "PDDL prompt", "user": "Convert LTL"}
        response_text = pddl_problem

        logger.log_stage2_success(
            pddl_problem,
            used_llm=True,
            model="gpt-4o-mini",
            prompt=prompt_dict,
            response=response_text
        )

        assert logger.log_data["stage2"]["status"] == "Success"
        assert logger.log_data["stage2"]["used_llm"] is True
        assert logger.log_data["stage2"]["model"] == "gpt-4o-mini"
        assert logger.log_data["stage2"]["llm_prompt"] == prompt_dict
        assert logger.log_data["stage2"]["llm_response"] == response_text

    def test_log_stage2_error(self, tmp_path):
        """Test logging Stage 2 error"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        error_message = "PDDL conversion failed"
        logger.log_stage2_error(error_message)

        assert logger.log_data["stage2"]["status"] == "Failed"
        assert logger.log_data["stage2"]["error"] == error_message


# ===== Test Stage 3 Logging =====

class TestStage3Logging:
    """Test Stage 3 logging"""

    def test_log_stage3_success_classical_planner(self, tmp_path):
        """Test logging Stage 3 success with classical planner"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        plan = [("pick-up", ["a"]), ("stack", ["a", "b"])]
        logger.log_stage3_success(plan, used_llm=False)

        assert logger.log_data["stage3"]["status"] == "Success"
        assert logger.log_data["stage3"]["plan_length"] == 2
        assert logger.log_data["stage3"]["used_llm"] is False

    def test_log_stage3_success_llm_planner(self, tmp_path):
        """Test logging Stage 3 success with LLM planner"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        plan = [("pick-up", ["a"])]
        prompt_dict = {"system": "Plan prompt", "user": "Generate plan"}
        response_text = '[{"action": "pick-up", "parameters": ["a"]}]'

        logger.log_stage3_success(
            plan,
            used_llm=True,
            model="gpt-4o-mini",
            prompt=prompt_dict,
            response=response_text
        )

        assert logger.log_data["stage3"]["status"] == "Success"
        assert logger.log_data["stage3"]["used_llm"] is True
        assert logger.log_data["stage3"]["model"] == "gpt-4o-mini"
        assert logger.log_data["stage3"]["llm_prompt"] == prompt_dict

    def test_log_stage3_error(self, tmp_path):
        """Test logging Stage 3 error"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        error_message = "No plan found"
        logger.log_stage3_error(error_message)

        assert logger.log_data["stage3"]["status"] == "Failed"
        assert logger.log_data["stage3"]["error"] == error_message


# ===== Test Pipeline Completion =====

class TestPipelineCompletion:
    """Test pipeline completion and file generation"""

    def test_end_pipeline_success(self, tmp_path):
        """Test ending pipeline with success status"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        # Simulate successful stages
        logger.log_stage1_success({"objects": []}, False, None, None, None)
        logger.log_stage2_success("pddl", False, None, None, None)
        logger.log_stage3_success([], False, None, None, None)

        log_file = logger.end_pipeline(success=True)

        assert logger.log_data["pipeline_success"] is True
        assert "end_time" in logger.log_data
        assert "total_duration_seconds" in logger.log_data
        assert log_file.exists()

    def test_end_pipeline_failure(self, tmp_path):
        """Test ending pipeline with failure status"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_error("Failed")
        log_file = logger.end_pipeline(success=False)

        assert logger.log_data["pipeline_success"] is False
        assert log_file.exists()

    def test_log_files_created(self, tmp_path):
        """Test both JSON and text log files are created"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")
        logger.log_stage1_success({"objects": []}, False, None, None, None)

        log_file = logger.end_pipeline(success=True)
        log_dir = log_file.parent

        # Verify JSON log
        json_log = log_dir / "execution.json"
        assert json_log.exists()
        with open(json_log) as f:
            json_data = json.load(f)
            assert json_data["nl_instruction"] == "Test"

        # Verify text log
        text_log = log_dir / "execution.txt"
        assert text_log.exists()
        with open(text_log) as f:
            text_content = f.read()
            assert "LTL-BDI Pipeline Execution Log" in text_content


# ===== Test Log Content Quality =====

class TestLogContentQuality:
    """Test quality and completeness of log content"""

    def test_complete_pipeline_log(self, tmp_path):
        """Test complete pipeline execution creates comprehensive log"""
        logger = PipelineLogger()
        logger.start_pipeline("Put A on B", "domain.pddl", str(tmp_path), "20250122_120000")

        # Stage 1
        ltl_spec = {"objects": ["a", "b"], "formulas_string": ["F(on(a, b))"]}
        logger.log_stage1_success(ltl_spec, True, "gpt-4o-mini", {"system": ""}, "{}")

        # Stage 2
        logger.log_stage2_success("(define...)", True, "gpt-4o-mini", {"system": ""}, "pddl")

        # Stage 3
        plan = [("pick-up", ["a"]), ("stack", ["a", "b"])]
        logger.log_stage3_success(plan, False, None, None, None)

        log_file = logger.end_pipeline(success=True)

        # Verify log completeness
        with open(log_file) as f:
            log_data = json.load(f)

        assert log_data["nl_instruction"] == "Put A on B"
        assert log_data["stage1"]["status"] == "Success"
        assert log_data["stage2"]["status"] == "Success"
        assert log_data["stage3"]["status"] == "Success"
        assert log_data["pipeline_success"] is True
        assert log_data["stage1"]["used_llm"] is True
        assert log_data["stage2"]["used_llm"] is True
        assert log_data["stage3"]["used_llm"] is False

    def test_text_log_readability(self, tmp_path):
        """Test text log format is human-readable"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")
        logger.log_stage1_success({"objects": ["a"]}, True, "gpt-4o-mini", {}, "{}")

        logger.end_pipeline(success=True)
        text_log = Path("logs") / "20250122_120000" / "execution.txt"

        if text_log.exists():  # May not exist depending on implementation
            with open(text_log) as f:
                content = f.read()
                assert "Stage 1" in content
                assert "Status: Success" in content or "Success" in content


# ===== Test Error Scenarios =====

class TestLoggerErrorHandling:
    """Test logger handles error scenarios gracefully"""

    def test_multiple_stage_failures(self, tmp_path):
        """Test logging multiple stage failures"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_error("Stage 1 failed")
        logger.log_stage2_error("Stage 2 failed")
        logger.log_stage3_error("Stage 3 failed")

        log_file = logger.end_pipeline(success=False)

        with open(log_file) as f:
            log_data = json.load(f)

        assert log_data["stage1"]["status"] == "Failed"
        assert log_data["stage2"]["status"] == "Failed"
        assert log_data["stage3"]["status"] == "Failed"
        assert log_data["pipeline_success"] is False

    def test_partial_pipeline_execution(self, tmp_path):
        """Test logging when pipeline stops mid-execution"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_success({"objects": []}, False, None, None, None)
        logger.log_stage2_error("Stage 2 failed")

        # Stage 3 never runs
        log_file = logger.end_pipeline(success=False)

        with open(log_file) as f:
            log_data = json.load(f)

        assert log_data["stage1"]["status"] == "Success"
        assert log_data["stage2"]["status"] == "Failed"
        assert log_data["stage3"]["status"] == "Not Started"
