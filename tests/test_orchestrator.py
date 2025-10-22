"""
Integration tests for Pipeline Orchestrator

Tests cover:
- End-to-end pipeline execution
- Stage coordination
- Error handling and recovery
- Logging and output generation
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestrator import PipelineOrchestrator
from stage1_interpretation.ltl_parser import LTLSpecification
from config import get_config


# ===== Test Orchestrator Initialization =====

class TestOrchestratorInitialization:
    """Test orchestrator initialization"""

    def test_initialization_with_default_domain(self, mock_env_with_api_key):
        """Test orchestrator initializes with default domain"""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.domain_file == "domains/blocksworld/domain.pddl"
        assert orchestrator.config is not None

    def test_initialization_with_custom_domain(self, mock_env_with_api_key):
        """Test orchestrator initializes with custom domain"""
        custom_domain = "domains/custom/domain.pddl"
        orchestrator = PipelineOrchestrator(domain_file=custom_domain)
        assert orchestrator.domain_file == custom_domain

    def test_components_initialized(self, mock_env_with_api_key):
        """Test all pipeline components are initialized"""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.ltl_parser is not None
        assert orchestrator.pddl_converter is not None
        assert orchestrator.classical_planner is not None


# ===== Test Pipeline Execution =====

@pytest.mark.integration
class TestPipelineExecution:
    """Test complete pipeline execution"""

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    @patch('orchestrator.PDDLPlanner')
    def test_execute_success_with_mocked_stages(
        self,
        mock_planner_class,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test successful pipeline execution with all stages mocked"""
        # Mock parser
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_spec.objects = ["a", "b"]
        mock_spec.initial_state = [{"ontable": ["a"]}]
        mock_spec.formulas = []
        mock_parser.parse.return_value = (mock_spec, {"system": "", "user": ""}, "{}")
        mock_parser_class.return_value = mock_parser

        # Mock converter
        mock_converter = MagicMock()
        mock_pddl = "(define (problem test) ...)"
        mock_converter.convert.return_value = (mock_pddl, {"system": "", "user": ""}, mock_pddl)
        mock_converter.get_constraints.return_value = []
        mock_converter_class.return_value = mock_converter

        # Mock planner
        mock_planner = MagicMock()
        mock_plan = [("pick-up", ["a"]), ("stack", ["a", "b"])]
        mock_planner.solve.return_value = mock_plan
        mock_planner_class.return_value = mock_planner

        orchestrator = PipelineOrchestrator()
        results = orchestrator.execute(
            nl_instruction="Put A on B",
            output_dir=str(tmp_path),
            enable_logging=False
        )

        # Verify all stages executed
        assert results["nl_instruction"] == "Put A on B"
        assert results["stage1_ltl"] is not None
        assert results["stage2_pddl"] is not None
        assert results["stage3_plan"] is not None

    @patch('orchestrator.NLToLTLParser')
    def test_execute_stage1_failure(
        self,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test pipeline handles Stage 1 failure correctly"""
        # Mock parser to raise error
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = RuntimeError("Stage 1 failed")
        mock_parser_class.return_value = mock_parser

        orchestrator = PipelineOrchestrator()

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.execute(
                nl_instruction="Test",
                output_dir=str(tmp_path),
                enable_logging=False
            )

        assert "Stage 1 failed" in str(exc_info.value)

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    def test_execute_stage2_failure(
        self,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test pipeline handles Stage 2 failure correctly"""
        # Mock parser success
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_parser.parse.return_value = (mock_spec, {}, "{}")
        mock_parser_class.return_value = mock_parser

        # Mock converter to raise error
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Stage 2 failed")
        mock_converter_class.return_value = mock_converter

        orchestrator = PipelineOrchestrator()

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.execute(
                nl_instruction="Test",
                output_dir=str(tmp_path),
                enable_logging=False
            )

        assert "Stage 2 failed" in str(exc_info.value)


# ===== Test Output Generation =====

@pytest.mark.integration
class TestOutputGeneration:
    """Test pipeline output file generation"""

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    @patch('orchestrator.PDDLPlanner')
    def test_output_files_created(
        self,
        mock_planner_class,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test pipeline creates all expected output files"""
        # Mock all stages
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_spec.objects = ["a"]
        mock_spec.initial_state = []
        mock_spec.formulas = []
        mock_parser.parse.return_value = (mock_spec, {}, "{}")
        mock_parser_class.return_value = mock_parser

        mock_converter = MagicMock()
        mock_pddl = "(define (problem test) ...)"
        mock_converter.convert.return_value = (mock_pddl, {}, mock_pddl)
        mock_converter.get_constraints.return_value = []
        mock_converter_class.return_value = mock_converter

        mock_planner = MagicMock()
        mock_plan = [("action", ["param"])]
        mock_planner.solve.return_value = mock_plan
        mock_planner_class.return_value = mock_planner

        orchestrator = PipelineOrchestrator()
        orchestrator.execute(
            nl_instruction="Test",
            output_dir=str(tmp_path),
            enable_logging=False
        )

        # Verify output directory was created with timestamp
        output_dirs = list(tmp_path.glob("*"))
        assert len(output_dirs) == 1  # One timestamped directory

        output_dir = output_dirs[0]
        assert (output_dir / "ltl_specification.json").exists()
        assert (output_dir / "problem.pddl").exists()
        assert (output_dir / "plan.txt").exists()

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    @patch('orchestrator.PDDLPlanner')
    def test_ltl_specification_format(
        self,
        mock_planner_class,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test LTL specification file contains correct format"""
        # Mock stages
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_spec.objects = ["a", "b"]
        mock_spec.initial_state = [{"ontable": ["a"]}]
        mock_spec.formulas = []
        mock_parser.parse.return_value = (mock_spec, {}, "{}")
        mock_parser_class.return_value = mock_parser

        mock_converter = MagicMock()
        mock_converter.convert.return_value = ("pddl", {}, "pddl")
        mock_converter.get_constraints.return_value = []
        mock_converter_class.return_value = mock_converter

        mock_planner = MagicMock()
        mock_planner.solve.return_value = [("action", [])]
        mock_planner_class.return_value = mock_planner

        orchestrator = PipelineOrchestrator()
        orchestrator.execute(
            nl_instruction="Test",
            output_dir=str(tmp_path),
            enable_logging=False
        )

        # Read LTL specification file
        output_dir = list(tmp_path.glob("*"))[0]
        ltl_file = output_dir / "ltl_specification.json"
        with open(ltl_file) as f:
            ltl_data = json.load(f)

        assert "objects" in ltl_data
        assert ltl_data["objects"] == ["a", "b"]


# ===== Test Logging Integration =====

@pytest.mark.integration
class TestLoggingIntegration:
    """Test pipeline logging functionality"""

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    @patch('orchestrator.PDDLPlanner')
    @patch('orchestrator.PipelineLogger')
    def test_logging_enabled(
        self,
        mock_logger_class,
        mock_planner_class,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        tmp_path
    ):
        """Test logging is enabled and captures all stages"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger.end_pipeline.return_value = tmp_path / "logs" / "execution.json"
        mock_logger_class.return_value = mock_logger

        # Mock stages
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_spec.objects = []
        mock_spec.initial_state = []
        mock_spec.formulas = []
        mock_parser.parse.return_value = (mock_spec, {}, "{}")
        mock_parser_class.return_value = mock_parser

        mock_converter = MagicMock()
        mock_converter.convert.return_value = ("pddl", {}, "pddl")
        mock_converter.get_constraints.return_value = []
        mock_converter_class.return_value = mock_converter

        mock_planner = MagicMock()
        mock_planner.solve.return_value = [("action", [])]
        mock_planner_class.return_value = mock_planner

        orchestrator = PipelineOrchestrator()
        orchestrator.execute(
            nl_instruction="Test",
            output_dir=str(tmp_path),
            enable_logging=True
        )

        # Verify logger methods were called
        mock_logger.start_pipeline.assert_called_once()
        mock_logger.log_stage1_success.assert_called_once()
        mock_logger.log_stage2_success.assert_called_once()
        mock_logger.log_stage3_success.assert_called_once()
        mock_logger.end_pipeline.assert_called_once_with(success=True)


# ===== Test Planner Selection =====

@pytest.mark.integration
class TestPlannerSelection:
    """Test orchestrator planner selection logic"""

    @patch('orchestrator.NLToLTLParser')
    @patch('orchestrator.LTLToPDDLConverter')
    @patch('orchestrator.LLMPlanner')
    def test_llm_planner_used_when_enabled(
        self,
        mock_llm_planner_class,
        mock_converter_class,
        mock_parser_class,
        mock_env_with_api_key,
        monkeypatch,
        tmp_path
    ):
        """Test LLM planner is used when USE_LLM_PLANNER=true"""
        monkeypatch.setenv("USE_LLM_PLANNER", "true")

        # Mock stages
        mock_parser = MagicMock()
        mock_spec = LTLSpecification()
        mock_spec.objects = []
        mock_spec.initial_state = []
        mock_spec.formulas = []
        mock_parser.parse.return_value = (mock_spec, {}, "{}")
        mock_parser_class.return_value = mock_parser

        mock_converter = MagicMock()
        mock_converter.convert.return_value = ("pddl", {}, "pddl")
        mock_converter.get_constraints.return_value = []
        mock_converter_class.return_value = mock_converter

        mock_llm_planner = MagicMock()
        mock_llm_planner.solve.return_value = ([("action", [])], {}, "{}")
        mock_llm_planner_class.return_value = mock_llm_planner

        orchestrator = PipelineOrchestrator()
        orchestrator.execute(
            nl_instruction="Test",
            output_dir=str(tmp_path),
            enable_logging=False
        )

        # Verify LLM planner was used
        mock_llm_planner.solve.assert_called_once()
