"""
Unit tests for Stage 2: LTL to PDDL Converter

Tests cover:
- PDDL problem generation
- LTL to PDDL conversion
- Constraint extraction (G operators)
- API key validation
- Error handling
- LLM integration
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
from stage2_translation.ltl_to_pddl import LTLToPDDLConverter


# ===== Helper Functions =====

def create_sample_ltl_spec():
    """Create a sample LTL specification for testing"""
    spec = LTLSpecification()
    spec.objects = ["a", "b"]
    spec.initial_state = [
        {"ontable": ["a"]},
        {"ontable": ["b"]},
        {"clear": ["a"]},
        {"clear": ["b"]},
        {"handempty": []}
    ]

    # F(on(a, b))
    atomic = LTLFormula(
        operator=None,
        predicate={"on": ["a", "b"]},
        sub_formulas=[],
        logical_op=None
    )
    formula = LTLFormula(
        operator=TemporalOperator.FINALLY,
        predicate=None,
        sub_formulas=[atomic],
        logical_op=None
    )
    spec.add_formula(formula)

    return spec


def create_ltl_spec_with_g_constraint():
    """Create LTL specification with G (Globally) constraint"""
    spec = create_sample_ltl_spec()

    # Add G(clear(c)) constraint
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
    spec.objects.append("c")

    return spec


# ===== Test LTLToPDDLConverter Class =====

class TestLTLToPDDLConverter:
    """Test LTL to PDDL converter"""

    def test_converter_initialization_with_api_key(self, mock_env_with_api_key):
        """Test converter initializes correctly with API key"""
        converter = LTLToPDDLConverter(
            api_key="sk-test-key",
            model="gpt-4o-mini",
            domain_name="blocksworld"
        )
        assert converter.api_key == "sk-test-key"
        assert converter.model == "gpt-4o-mini"
        assert converter.domain_name == "blocksworld"
        assert converter.client is not None

    def test_converter_initialization_without_api_key(self):
        """Test converter initializes without client when no API key"""
        converter = LTLToPDDLConverter(api_key=None)
        assert converter.client is None

    def test_convert_raises_error_without_api_key(self):
        """Test convert raises RuntimeError when no API key configured"""
        converter = LTLToPDDLConverter(api_key=None)
        ltl_spec = create_sample_ltl_spec()

        with pytest.raises(RuntimeError) as exc_info:
            converter.convert("test_problem", ltl_spec)

        assert "No API key configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    @pytest.mark.unit
    def test_convert_with_llm_success(self, mock_env_with_api_key, mock_llm_pddl_response):
        """Test successful conversion with LLM"""
        converter = LTLToPDDLConverter(
            api_key="sk-test-key",
            model="gpt-4o-mini",
            domain_name="blocksworld"
        )
        ltl_spec = create_sample_ltl_spec()

        # Mock OpenAI client response
        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_pddl_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, prompt_dict, response_text = converter.convert(
                "test_problem",
                ltl_spec
            )

            # Verify PDDL structure
            assert pddl_problem.startswith("(define (problem")
            assert "blocksworld" in pddl_problem
            assert ":objects" in pddl_problem
            assert ":init" in pddl_problem
            assert ":goal" in pddl_problem

            # Verify prompt was created
            assert "system" in prompt_dict
            assert "user" in prompt_dict
            assert "test_problem" in prompt_dict["user"]

            # Verify response was captured
            assert response_text == mock_llm_pddl_response

            # Verify LLM was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-4o-mini"
            assert call_args.kwargs["temperature"] == 0.0

    @pytest.mark.unit
    def test_convert_with_domain_file(self, mock_env_with_api_key, mock_llm_pddl_response, tmp_path):
        """Test conversion with domain file context"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        # Create a temporary domain file
        domain_file = tmp_path / "domain.pddl"
        domain_file.write_text("(define (domain blocksworld) ...)")

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_pddl_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, prompt_dict, _ = converter.convert(
                "test_problem",
                ltl_spec,
                domain_file_path=str(domain_file)
            )

            # Verify domain context was included in prompt
            assert "Domain Context:" in prompt_dict["system"]

    @pytest.mark.unit
    def test_convert_with_llm_api_error(self, mock_env_with_api_key):
        """Test convert handles LLM API errors correctly"""
        converter = LTLToPDDLConverter(api_key="sk-test-key", model="gpt-4o-mini")
        ltl_spec = create_sample_ltl_spec()

        with patch.object(converter, 'client') as mock_client:
            # Simulate API error
            mock_client.chat.completions.create.side_effect = Exception("API timeout")

            with pytest.raises(RuntimeError) as exc_info:
                converter.convert("test_problem", ltl_spec)

            assert "LLM PDDL conversion failed" in str(exc_info.value)
            assert "API timeout" in str(exc_info.value)
            assert "test_problem" in str(exc_info.value)
            assert "gpt-4o-mini" in str(exc_info.value)

    @pytest.mark.unit
    def test_convert_with_invalid_pddl_response(self, mock_env_with_api_key):
        """Test convert handles invalid PDDL response from LLM"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "This is not valid PDDL"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                converter.convert("test_problem", ltl_spec)

            assert "not valid PDDL problem format" in str(exc_info.value)

    @pytest.mark.unit
    def test_convert_with_markdown_wrapped_pddl(self, mock_env_with_api_key):
        """Test convert handles markdown-wrapped PDDL response"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        markdown_wrapped = """```
(define (problem test_problem)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (ontable a) (ontable b))
  (:goal (on a b))
)
```"""

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = markdown_wrapped
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, _, _ = converter.convert("test_problem", ltl_spec)

            # Verify markdown was stripped
            assert pddl_problem.startswith("(define (problem")
            assert "```" not in pddl_problem


# ===== Test Constraint Extraction =====

class TestConstraintExtraction:
    """Test extraction of G (Globally) constraints"""

    def test_get_constraints_empty(self):
        """Test constraint extraction with no G operators"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()  # Only has F operator

        constraints = converter.get_constraints(ltl_spec)
        assert len(constraints) == 0

    def test_get_constraints_with_g_operator(self):
        """Test constraint extraction with G operator"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_ltl_spec_with_g_constraint()

        constraints = converter.get_constraints(ltl_spec)
        assert len(constraints) == 1
        assert constraints[0]["type"] == "globally"
        assert constraints[0]["predicate"] == {"clear": ["c"]}
        assert "G(" in constraints[0]["formula_string"]

    def test_get_constraints_multiple_g_operators(self):
        """Test constraint extraction with multiple G operators"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        # Add G(clear(a))
        atomic1 = LTLFormula(
            operator=None,
            predicate={"clear": ["a"]},
            sub_formulas=[],
            logical_op=None
        )
        formula1 = LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[atomic1],
            logical_op=None
        )
        ltl_spec.add_formula(formula1)

        # Add G(handempty)
        atomic2 = LTLFormula(
            operator=None,
            predicate={"handempty": []},
            sub_formulas=[],
            logical_op=None
        )
        formula2 = LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[atomic2],
            logical_op=None
        )
        ltl_spec.add_formula(formula2)

        constraints = converter.get_constraints(ltl_spec)
        assert len(constraints) == 2
        assert all(c["type"] == "globally" for c in constraints)


# ===== Test PDDL Generation Quality =====

class TestPDDLGenerationQuality:
    """Test quality and correctness of generated PDDL"""

    @pytest.mark.unit
    def test_pddl_contains_all_objects(self, mock_env_with_api_key, mock_llm_pddl_response):
        """Test generated PDDL contains all objects from LTL spec"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_pddl_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, _, _ = converter.convert("test_problem", ltl_spec)

            # Verify all objects are in PDDL
            for obj in ltl_spec.objects:
                assert obj in pddl_problem

    @pytest.mark.unit
    def test_pddl_contains_initial_state(self, mock_env_with_api_key, mock_llm_pddl_response):
        """Test generated PDDL contains initial state predicates"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_pddl_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, _, _ = converter.convert("test_problem", ltl_spec)

            # Verify :init section exists
            assert ":init" in pddl_problem
            assert "ontable" in pddl_problem
            assert "clear" in pddl_problem
            assert "handempty" in pddl_problem

    @pytest.mark.unit
    def test_pddl_contains_goal(self, mock_env_with_api_key, mock_llm_pddl_response):
        """Test generated PDDL contains goal from LTL formulas"""
        converter = LTLToPDDLConverter(api_key="sk-test-key")
        ltl_spec = create_sample_ltl_spec()

        with patch.object(converter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = mock_llm_pddl_response
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            pddl_problem, _, _ = converter.convert("test_problem", ltl_spec)

            # Verify :goal section exists
            assert ":goal" in pddl_problem
            assert "on a b" in pddl_problem or "(on a b)" in pddl_problem


# ===== Integration Marker Tests =====

@pytest.mark.integration
@pytest.mark.requires_api_key
class TestLTLToPDDLConverterIntegration:
    """Integration tests requiring actual API key (marked for selective running)"""

    def test_convert_with_real_api(self, mock_env_with_api_key):
        """Test conversion with real OpenAI API (skipped in CI without API key)"""
        import os
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-test"):
            pytest.skip("Skipping real API test - no valid API key")

        converter = LTLToPDDLConverter(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            domain_name="blocksworld"
        )
        ltl_spec = create_sample_ltl_spec()

        pddl_problem, prompt, response = converter.convert("test_problem", ltl_spec)

        assert pddl_problem.startswith("(define (problem")
        assert "blocksworld" in pddl_problem
        assert len(pddl_problem) > 100
