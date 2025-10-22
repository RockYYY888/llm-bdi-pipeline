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

    def test_logger_creates_log_directory(self, tmp_path):
        """Test logger creates log directory"""
        log_dir = tmp_path / "logs"
        logger = PipelineLogger(logs_dir=str(log_dir))
        assert logger is not None
        assert logger.logs_dir.exists()

    def test_start_pipeline(self, tmp_path):
        """Test start_pipeline initializes record"""
        logger = PipelineLogger()
        logger.start_pipeline(
            natural_language="Test instruction",
            domain_file="domain.pddl",
            output_dir=str(tmp_path),
            timestamp="20250122_120000"
        )

        assert logger.current_record.natural_language == "Test instruction"
        assert logger.current_record.domain_file == "domain.pddl"
        assert logger.current_record.output_dir == str(tmp_path)
        assert logger.current_record.timestamp == "20250122_120000"
        assert logger.start_time is not None


# ===== Test Stage Logging =====

class TestStageLogging:
    """Test stage logging functionality"""

    def test_log_stage1_success(self, tmp_path):
        """Test logging Stage 1 success"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        ltl_spec_dict = {"objects": ["a"], "formulas_string": ["F(on(a, b))"]}

        logger.log_stage1_success(
            ltl_spec_dict,
            used_llm=True,
            model="gpt-4o-mini",
            llm_prompt={"system": "test", "user": "test"},
            llm_response="{}"
        )

        assert logger.current_record.stage1_status == "success"
        assert logger.current_record.stage1_ltl_spec == ltl_spec_dict
        assert logger.current_record.stage1_used_llm is True

    def test_log_stage1_error(self, tmp_path):
        """Test logging Stage 1 error"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_error("Error message")

        assert logger.current_record.stage1_status == "failed"
        assert logger.current_record.stage1_error == "Error message"

    def test_log_stage2_success(self, tmp_path):
        """Test logging Stage 2 success"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage2_success(
            "(define (problem test) ...)",
            used_llm=True,
            model="gpt-4o-mini",
            llm_prompt={"system": "test"},
            llm_response="pddl"
        )

        assert logger.current_record.stage2_status == "success"
        assert logger.current_record.stage2_used_llm is True

    def test_log_stage3_success(self, tmp_path):
        """Test logging Stage 3 success"""
        logger = PipelineLogger()
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        plan = [("pick-up", ["a"]), ("stack", ["a", "b"])]
        logger.log_stage3_success(plan)

        assert logger.current_record.stage3_status == "success"
        assert logger.current_record.stage3_plan == plan


# ===== Test Pipeline Completion =====

class TestPipelineCompletion:
    """Test pipeline completion and file generation"""

    def test_end_pipeline_success(self, tmp_path):
        """Test ending pipeline with success"""
        logger = PipelineLogger(logs_dir=str(tmp_path / "logs"))
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_success({"objects": []}, False)

        log_file = logger.end_pipeline(success=True)

        assert log_file.exists()
        assert logger.current_record.success is True

    def test_end_pipeline_failure(self, tmp_path):
        """Test ending pipeline with failure"""
        logger = PipelineLogger(logs_dir=str(tmp_path / "logs"))
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_error("Failed")
        log_file = logger.end_pipeline(success=False)

        assert log_file.exists()
        assert logger.current_record.success is False

    def test_log_files_created(self, tmp_path):
        """Test both JSON and text log files are created"""
        logger = PipelineLogger(logs_dir=str(tmp_path / "logs"))
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")
        logger.log_stage1_success({"objects": []}, False)

        log_file = logger.end_pipeline(success=True)
        log_dir = log_file.parent

        # Verify JSON log
        json_log = log_dir / "execution.json"
        assert json_log.exists()

        # Verify text log
        text_log = log_dir / "execution.txt"
        assert text_log.exists()


# ===== Test Error Scenarios =====

class TestErrorScenarios:
    """Test logger handles error scenarios"""

    def test_multiple_stage_failures(self, tmp_path):
        """Test logging multiple stage failures"""
        logger = PipelineLogger(logs_dir=str(tmp_path / "logs"))
        logger.start_pipeline("Test", "domain.pddl", str(tmp_path), "20250122_120000")

        logger.log_stage1_error("Stage 1 failed")
        logger.log_stage2_error("Stage 2 failed")
        logger.log_stage3_error("Stage 3 failed")

        log_file = logger.end_pipeline(success=False)

        with open(log_file) as f:
            log_data = json.load(f)

        assert log_data["stage1_status"] == "failed"
        assert log_data["stage2_status"] == "failed"
        assert log_data["stage3_status"] == "failed"
        assert log_data["success"] is False
