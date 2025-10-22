"""
Unit tests for Stage 1: Natural Language to LTL Parser

Tests cover:
- LTL formula parsing and structure
- Temporal operators (F, G, X, U)
- Nested operators (F(G(φ)), G(F(φ)))
- API key validation
- Error handling
- LLM integration
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from stage1_interpretation.ltl_parser import (
    NLToLTLParser,
    LTLSpecification,
    LTLFormula,
    TemporalOperator,
    LogicalOperator
)


# ===== Test LTLFormula Class =====

class TestLTLFormula:
    """Test LTL formula structure and string representation"""

    def test_atomic_formula_to_string(self):
        """Test atomic predicate conversion to string"""
        formula = LTLFormula(
            operator=None,
            predicate={"on": ["a", "b"]},
            sub_formulas=[],
            logical_op=None
        )
        assert formula.to_string() == "on(a, b)"

    def test_finally_operator_to_string(self):
        """Test F (Finally) operator string representation"""
        atomic = LTLFormula(
            operator=None,
            predicate={"clear": ["a"]},
            sub_formulas=[],
            logical_op=None
        )
        formula = LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[atomic],
            logical_op=None
        )
        assert formula.to_string() == "F(clear(a))"

    def test_globally_operator_to_string(self):
        """Test G (Globally) operator string representation"""
        atomic = LTLFormula(
            operator=None,
            predicate={"ontable": ["b"]},
            sub_formulas=[],
            logical_op=None
        )
        formula = LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[atomic],
            logical_op=None
        )
        assert formula.to_string() == "G(ontable(b))"

    def test_next_operator_to_string(self):
        """Test X (Next) operator string representation"""
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
        assert formula.to_string() == "X(holding(a))"

    def test_until_operator_to_string(self):
        """Test U (Until) operator string representation"""
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
        assert formula.to_string() == "(holding(a) U clear(b))"

    def test_nested_fg_operator_to_string(self):
        """Test F(G(φ)) nested operator string representation"""
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
        assert outer.to_string() == "F(G(on(a, b)))"

    def test_nested_gf_operator_to_string(self):
        """Test G(F(φ)) nested operator string representation"""
        atomic = LTLFormula(
            operator=None,
            predicate={"clear": ["c"]},
            sub_formulas=[],
            logical_op=None
        )
        inner = LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[atomic],
            logical_op=None
        )
        outer = LTLFormula(
            operator=TemporalOperator.GLOBALLY,
            predicate=None,
            sub_formulas=[inner],
            logical_op=None
        )
        assert outer.to_string() == "G(F(clear(c)))"

    def test_logical_and_to_string(self):
        """Test logical AND operator string representation"""
        f1 = LTLFormula(
            operator=None,
            predicate={"on": ["a", "b"]},
            sub_formulas=[],
            logical_op=None
        )
        f2 = LTLFormula(
            operator=None,
            predicate={"clear": ["a"]},
            sub_formulas=[],
            logical_op=None
        )
        formula = LTLFormula(
            operator=None,
            predicate=None,
            sub_formulas=[f1, f2],
            logical_op=LogicalOperator.AND
        )
        assert formula.to_string() == "(on(a, b) & clear(a))"

    def test_formula_to_dict(self):
        """Test LTL formula serialization to dictionary"""
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
        result = formula.to_dict()
        assert result["operator"] == "F"
        assert result["predicate"] is None
        assert result["logical_op"] is None
        assert len(result["sub_formulas"]) == 1
        assert result["sub_formulas"][0]["predicate"] == {"on": ["a", "b"]}


# ===== Test LTLSpecification Class =====

class TestLTLSpecification:
    """Test LTL specification structure"""

    def test_add_formula(self):
        """Test adding formulas to specification"""
        spec = LTLSpecification()
        formula = LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[],
            logical_op=None
        )
        spec.add_formula(formula)
        assert len(spec.formulas) == 1
        assert spec.formulas[0] == formula

    def test_to_dict(self):
        """Test specification serialization to dictionary"""
        spec = LTLSpecification()
        spec.objects = ["a", "b"]
        spec.initial_state = [{"ontable": ["a"]}]

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

        result = spec.to_dict()
        assert result["objects"] == ["a", "b"]
        assert result["initial_state"] == [{"ontable": ["a"]}]
        assert len(result["formulas"]) == 1
        assert result["formulas_string"] == ["F(on(a, b))"]


# ===== Test NLToLTLParser Class =====

class TestNLToLTLParser:
    """Test natural language to LTL parser"""

    def test_parser_initialization_with_api_key(self, mock_env_with_api_key):
        """Test parser initializes correctly with API key"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")
        assert parser.api_key == "sk-test-key"
        assert parser.model == "gpt-4o-mini"
        assert parser.client is not None

    def test_parser_initialization_without_api_key(self):
        """Test parser initializes without client when no API key"""
        parser = NLToLTLParser(api_key=None)
        assert parser.client is None

    def test_parse_raises_error_without_api_key(self):
        """Test parse raises RuntimeError when no API key configured"""
        parser = NLToLTLParser(api_key=None)
        with pytest.raises(RuntimeError) as exc_info:
            parser.parse("Put block A on block B")
        assert "No API key configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    @pytest.mark.unit
    def test_parse_with_llm_success(self, mock_env_with_api_key, mock_llm_ltl_response):
        """Test successful parsing with LLM"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        # Mock OpenAI client response
        with patch.object(parser, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = json.dumps(mock_llm_ltl_response)
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            spec, prompt_dict, response_text = parser.parse("Put block A on block B")

            # Verify result structure
            assert isinstance(spec, LTLSpecification)
            assert spec.objects == ["a", "b"]
            assert len(spec.initial_state) == 5
            assert len(spec.formulas) == 2

            # Verify prompt was created
            assert "system" in prompt_dict
            assert "user" in prompt_dict
            assert "Put block A on block B" in prompt_dict["user"]

            # Verify response was captured
            assert response_text == json.dumps(mock_llm_ltl_response)

            # Verify LLM was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-4o-mini"
            assert call_args.kwargs["temperature"] == 0.0

    @pytest.mark.unit
    def test_parse_with_llm_api_error(self, mock_env_with_api_key):
        """Test parse handles LLM API errors correctly"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        with patch.object(parser, 'client') as mock_client:
            # Simulate API error
            mock_client.chat.completions.create.side_effect = Exception("API connection failed")

            with pytest.raises(RuntimeError) as exc_info:
                parser.parse("Put block A on block B")

            assert "LLM API call failed" in str(exc_info.value)
            assert "API connection failed" in str(exc_info.value)
            assert "gpt-4o-mini" in str(exc_info.value)

    @pytest.mark.unit
    def test_parse_with_invalid_json_response(self, mock_env_with_api_key):
        """Test parse handles invalid JSON response from LLM"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        with patch.object(parser, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "This is not valid JSON"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                parser.parse("Put block A on block B")

            assert "Failed to parse LLM response as JSON" in str(exc_info.value)

    @pytest.mark.unit
    def test_parse_temporal_operators(self, mock_env_with_api_key):
        """Test parsing all temporal operators (F, G, X)"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        test_response = {
            "objects": ["a", "b", "c"],
            "initial_state": [{"ontable": ["a"]}, {"handempty": []}],
            "ltl_formulas": [
                {"type": "temporal", "operator": "F", "formula": {"on": ["a", "b"]}},
                {"type": "temporal", "operator": "G", "formula": {"clear": ["c"]}},
                {"type": "temporal", "operator": "X", "formula": {"holding": ["a"]}}
            ]
        }

        with patch.object(parser, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = json.dumps(test_response)
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            spec, _, _ = parser.parse("Test instruction")

            assert len(spec.formulas) == 3
            assert spec.formulas[0].operator == TemporalOperator.FINALLY
            assert spec.formulas[1].operator == TemporalOperator.GLOBALLY
            assert spec.formulas[2].operator == TemporalOperator.NEXT

    @pytest.mark.unit
    def test_parse_until_operator(self, mock_env_with_api_key):
        """Test parsing Until (U) operator"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        test_response = {
            "objects": ["a", "b"],
            "initial_state": [{"handempty": []}],
            "ltl_formulas": [
                {
                    "type": "until",
                    "operator": "U",
                    "left_formula": {"holding": ["a"]},
                    "right_formula": {"clear": ["b"]}
                }
            ]
        }

        with patch.object(parser, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = json.dumps(test_response)
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            spec, _, _ = parser.parse("Hold A until B is clear")

            assert len(spec.formulas) == 1
            assert spec.formulas[0].operator == TemporalOperator.UNTIL
            assert len(spec.formulas[0].sub_formulas) == 2
            assert spec.formulas[0].to_string() == "(holding(a) U clear(b))"

    @pytest.mark.unit
    def test_parse_nested_operators(self, mock_env_with_api_key):
        """Test parsing nested operators (F(G(φ)), G(F(φ)))"""
        parser = NLToLTLParser(api_key="sk-test-key", model="gpt-4o-mini")

        test_response = {
            "objects": ["a", "b"],
            "initial_state": [{"handempty": []}],
            "ltl_formulas": [
                {
                    "type": "nested",
                    "outer_operator": "F",
                    "inner_operator": "G",
                    "formula": {"on": ["a", "b"]}
                },
                {
                    "type": "nested",
                    "outer_operator": "G",
                    "inner_operator": "F",
                    "formula": {"clear": ["a"]}
                }
            ]
        }

        with patch.object(parser, 'client') as mock_client:
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = json.dumps(test_response)
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            spec, _, _ = parser.parse("Eventually ensure A is always on B")

            assert len(spec.formulas) == 2
            assert spec.formulas[0].to_string() == "F(G(on(a, b)))"
            assert spec.formulas[1].to_string() == "G(F(clear(a)))"


# ===== Integration Marker Tests =====

@pytest.mark.integration
@pytest.mark.requires_api_key
class TestNLToLTLParserIntegration:
    """Integration tests requiring actual API key (marked for selective running)"""

    def test_parse_with_real_api(self, mock_env_with_api_key):
        """Test parsing with real OpenAI API (skipped in CI without API key)"""
        import os
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-test"):
            pytest.skip("Skipping real API test - no valid API key")

        parser = NLToLTLParser(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )
        spec, prompt, response = parser.parse("Put block A on block B")

        assert isinstance(spec, LTLSpecification)
        assert len(spec.objects) > 0
        assert len(spec.formulas) > 0
