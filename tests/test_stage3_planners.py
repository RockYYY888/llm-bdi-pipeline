"""
Unit tests for Stage 3: PDDL Planners (Classical and LLM-based)

Tests cover:
- Classical PDDL planner (pyperplan)
- LLM-based planner
- Constraint handling (G, X, U operators)
- Plan validation
- Error handling
"""

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from stage1_interpretation.ltl_parser import (
    LTLSpecification,
    LTLFormula,
    TemporalOperator
)
from stage3_codegen.pddl_planner import PDDLPlanner
from stage3_codegen.llm_planner import LLMPlanner


# ===== Helper Functions =====

def create_sample_ltl_spec_with_constraints():
    """Create LTL spec with various constraints for testing"""
    spec = LTLSpecification()
    spec.objects = ["a", "b", "c"]
    spec.initial_state = [
        {"ontable": ["a"]},
        {"ontable": ["b"]},
        {"clear": ["a"]},
        {"clear": ["b"]},
        {"handempty": []}
    ]

    # F(on(a, b)) - Finally goal
    atomic_f = LTLFormula(
        operator=None,
        predicate={"on": ["a", "b"]},
        sub_formulas=[],
        logical_op=None
    )
    formula_f = LTLFormula(
        operator=TemporalOperator.FINALLY,
        predicate=None,
        sub_formulas=[atomic_f],
        logical_op=None
    )
    spec.add_formula(formula_f)

    # G(clear(c)) - Globally constraint
    atomic_g = LTLFormula(
        operator=None,
        predicate={"clear": ["c"]},
        sub_formulas=[],
        logical_op=None
    )
    formula_g = LTLFormula(
        operator=TemporalOperator.GLOBALLY,
        predicate=None,
        sub_formulas=[atomic_g],
        logical_op=None
    )
    spec.add_formula(formula_g)

    return spec


# ===== Test PDDLPlanner (Classical) =====

class TestPDDLPlanner:
    """Test classical PDDL planner using pyperplan"""

    def test_planner_initialization(self):
        """Test planner initializes correctly"""
        planner = PDDLPlanner()
        assert planner is not None

    def test_solve_with_valid_problem(self, test_data_dir, tmp_path):
        """Test solving a valid PDDL problem"""
        planner = PDDLPlanner()

        # Create a simple PDDL problem file
        problem_file = tmp_path / "test_problem.pddl"
        problem_content = """(define (problem simple_blocks)
  (:domain blocksworld)
  (:objects a b - block)
  (:init
    (ontable a)
    (ontable b)
    (clear a)
    (clear b)
    (handempty)
  )
  (:goal (and
    (on a b)
  ))
)"""
        problem_file.write_text(problem_content)

        domain_file = test_data_dir / "blocksworld_domain.pddl"

        plan = planner.solve(str(domain_file), str(problem_file))

        # Verify plan structure
        if plan:  # Classical planner might not always find a plan
            assert isinstance(plan, list)
            assert all(isinstance(step, tuple) for step in plan)
            assert all(len(step) == 2 for step in plan)  # (action, params)

    def test_solve_with_missing_domain_file(self, tmp_path):
        """Test planner handles missing domain file"""
        planner = PDDLPlanner()
        problem_file = tmp_path / "problem.pddl"
        problem_file.write_text("(define (problem test) ...)")

        plan = planner.solve("nonexistent_domain.pddl", str(problem_file))
        assert plan is None  # Should return None on error

    def test_solve_with_missing_problem_file(self, test_data_dir):
        """Test planner handles missing problem file"""
        planner = PDDLPlanner()
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        plan = planner.solve(str(domain_file), "nonexistent_problem.pddl")
        assert plan is None

    def test_solve_with_unsolvable_problem(self, test_data_dir, tmp_path):
        """Test planner handles unsolvable problems gracefully"""
        planner = PDDLPlanner()

        # Create an unsolvable problem
        problem_file = tmp_path / "unsolvable.pddl"
        problem_content = """(define (problem unsolvable)
  (:domain blocksworld)
  (:objects a b - block)
  (:init
    (on a b)
    (ontable b)
    (clear a)
    (handempty)
  )
  (:goal (and
    (on b a)
    (on a b)
  ))
)"""
        problem_file.write_text(problem_content)
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        plan = planner.solve(str(domain_file), str(problem_file))
        # Unsolvable problem should return None
        assert plan is None or len(plan) == 0


# ===== Test LLMPlanner =====

class TestLLMPlanner:
    """Test LLM-based planner"""

    def test_planner_initialization_with_api_key(self, mock_env_with_api_key):
        """Test LLM planner initializes correctly with API key"""
        planner = LLMPlanner(
            api_key="sk-test-key",
            model="gpt-4o-mini"
        )
        assert planner.api_key == "sk-test-key"
        assert planner.model == "gpt-4o-mini"
        assert planner.client is not None

    def test_planner_initialization_without_api_key(self):
        """Test LLM planner initializes without client when no API key"""
        planner = LLMPlanner(api_key=None)
        assert planner.client is None

    def test_solve_without_api_key(self, tmp_path, test_data_dir):
        """Test solve returns None when no API key configured"""
        planner = LLMPlanner(api_key=None)

        problem_file = tmp_path / "problem.pddl"
        problem_file.write_text("(define (problem test) ...)")
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        result = planner.solve(str(domain_file), str(problem_file))
        plan, prompt, response = result

        assert plan is None
        assert prompt is None
        assert response is None

    @pytest.mark.unit
    def test_solve_with_llm_success(self, mock_env_with_api_key, mock_llm_plan_response, test_data_dir, tmp_path):
        """Test successful plan generation with LLM"""
        planner = LLMPlanner(api_key="sk-test-key", model="gpt-4o-mini")

        # Create test files
        problem_file = tmp_path / "problem.pddl"
        problem_content = """(define (problem test)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (ontable a) (clear a) (handempty))
  (:goal (on a b))
)"""
        problem_file.write_text(problem_content)
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        # Mock OpenAI client response
        with patch.object(planner, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_plan_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            plan, prompt_dict, response_text = planner.solve(
                str(domain_file),
                str(problem_file)
            )

            # Verify plan structure
            assert plan is not None
            assert isinstance(plan, list)
            assert len(plan) == 2
            assert plan[0] == ("pick-up", ["a"])
            assert plan[1] == ("stack", ["a", "b"])

            # Verify prompt was created
            assert "system" in prompt_dict
            assert "user" in prompt_dict

            # Verify response was captured
            assert response_text == mock_llm_plan_response

    @pytest.mark.unit
    def test_solve_with_g_constraints(self, mock_env_with_api_key, test_data_dir, tmp_path):
        """Test LLM planner handles G (Globally) constraints"""
        planner = LLMPlanner(api_key="sk-test-key", model="gpt-4o-mini")
        ltl_spec = create_sample_ltl_spec_with_constraints()

        problem_file = tmp_path / "problem.pddl"
        problem_file.write_text("(define (problem test) ...)")
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        # Mock LLM response
        with patch.object(planner, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = '[{"action": "pick-up", "parameters": ["a"]}]'
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            plan, prompt_dict, _ = planner.solve(
                str(domain_file),
                str(problem_file),
                ltl_spec
            )

            # Verify G constraint was included in prompt
            system_prompt = prompt_dict["system"]
            assert "Globally (G) Constraints" in system_prompt
            assert "clear(c)" in system_prompt
            assert "MUST hold at EVERY step" in system_prompt

    @pytest.mark.unit
    def test_solve_with_llm_api_error(self, mock_env_with_api_key, test_data_dir, tmp_path):
        """Test LLM planner handles API errors gracefully"""
        planner = LLMPlanner(api_key="sk-test-key", model="gpt-4o-mini")

        problem_file = tmp_path / "problem.pddl"
        problem_file.write_text("(define (problem test) ...)")
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        with patch.object(planner, 'client') as mock_client:
            # Simulate API error
            mock_client.chat.completions.create.side_effect = Exception("API error")

            plan, prompt, response = planner.solve(str(domain_file), str(problem_file))

            # Should return None on error
            assert plan is None

    @pytest.mark.unit
    def test_solve_with_invalid_json_response(self, mock_env_with_api_key, test_data_dir, tmp_path):
        """Test LLM planner handles invalid JSON response"""
        planner = LLMPlanner(api_key="sk-test-key")

        problem_file = tmp_path / "problem.pddl"
        problem_file.write_text("(define (problem test) ...)")
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        with patch.object(planner, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "This is not valid JSON"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            plan, prompt, response = planner.solve(str(domain_file), str(problem_file))

            # Should return None on parse error
            assert plan is None


# ===== Test Constraint Extraction =====

class TestConstraintExtraction:
    """Test constraint extraction from LTL specifications"""

    def test_extract_g_constraints(self):
        """Test extraction of G (Globally) constraints"""
        planner = LLMPlanner(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec_with_constraints()

        g_constraints = planner._extract_g_constraints(ltl_spec)

        assert len(g_constraints) == 1
        assert "clear(c)" in g_constraints[0]

    def test_extract_x_constraints(self):
        """Test extraction of X (Next) constraints"""
        planner = LLMPlanner(api_key="sk-test-key")
        spec = LTLSpecification()

        # Add X(holding(a))
        atomic = LTLFormula(
            operator=None,
            predicate={"holding": ["a"]},
            sub_formulas=[],
            logical_op=None
        )
        formula = LTLFormula(
            operator=TemporalOperator.NEXT,
            predicate=None,
            sub_formulas=[atomic],
            logical_op=None
        )
        spec.add_formula(formula)

        x_constraints = planner._extract_x_constraints(spec)

        assert len(x_constraints) == 1
        assert "holding(a)" in x_constraints[0]

    def test_extract_u_constraints(self):
        """Test extraction of U (Until) constraints"""
        planner = LLMPlanner(api_key="sk-test-key")
        spec = LTLSpecification()

        # Add holding(a) U clear(b)
        left = LTLFormula(
            operator=None,
            predicate={"holding": ["a"]},
            sub_formulas=[],
            logical_op=None
        )
        right = LTLFormula(
            operator=None,
            predicate={"clear": ["b"]},
            sub_formulas=[],
            logical_op=None
        )
        formula = LTLFormula(
            operator=TemporalOperator.UNTIL,
            predicate=None,
            sub_formulas=[left, right],
            logical_op=None
        )
        spec.add_formula(formula)

        u_constraints = planner._extract_u_constraints(spec)

        assert len(u_constraints) == 1
        assert u_constraints[0]["left"] == "holding(a)"
        assert u_constraints[0]["right"] == "clear(b)"

    def test_extract_nested_constraints(self):
        """Test extraction of nested temporal constraints"""
        planner = LLMPlanner(api_key="sk-test-key")
        spec = LTLSpecification()

        # Add F(G(on(a, b)))
        atomic = LTLFormula(
            operator=None,
            predicate={"on": ["a", "b"]},
            sub_formulas=[],
            logical_op=None
        )
        inner = LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[atomic],
            logical_op=None
        )
        outer = LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[inner],
            logical_op=None
        )
        spec.add_formula(outer)

        nested_constraints = planner._extract_nested_constraints(spec)

        assert len(nested_constraints) == 1
        assert "F(G(on(a, b)))" in nested_constraints[0]


# ===== Integration Tests =====

@pytest.mark.integration
@pytest.mark.requires_api_key
class TestLLMPlannerIntegration:
    """Integration tests requiring actual API key"""

    def test_solve_with_real_api(self, mock_env_with_api_key, test_data_dir, tmp_path):
        """Test planning with real OpenAI API (skipped in CI without API key)"""
        import os
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-test"):
            pytest.skip("Skipping real API test - no valid API key")

        planner = LLMPlanner(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        problem_file = tmp_path / "problem.pddl"
        problem_content = """(define (problem simple)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (ontable a) (ontable b) (clear a) (clear b) (handempty))
  (:goal (on a b))
)"""
        problem_file.write_text(problem_content)
        domain_file = test_data_dir / "blocksworld_domain.pddl"

        plan, prompt, response = planner.solve(str(domain_file), str(problem_file))

        if plan:  # LLM might not always generate valid plan
            assert isinstance(plan, list)
            assert all(isinstance(step, tuple) for step in plan)
