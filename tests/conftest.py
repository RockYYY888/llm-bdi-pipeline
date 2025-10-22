"""
Pytest configuration and shared fixtures for LLM-BDI Pipeline tests

This file provides common test fixtures, mocks, and utilities used across all tests.
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock


# ===== Test Data Directory =====

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Returns the test data directory path"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def domain_file(test_data_dir: Path) -> str:
    """Returns path to test PDDL domain file"""
    return str(test_data_dir / "blocksworld_domain.pddl")


# ===== Environment Configuration Fixtures =====

@pytest.fixture
def mock_env_with_api_key(monkeypatch):
    """Set up environment with mock API key"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-12345")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    yield
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.fixture
def mock_env_no_api_key(monkeypatch):
    """Set up environment without API key (should trigger errors)"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


# ===== Mock LLM Response Fixtures =====

@pytest.fixture
def mock_llm_ltl_response() -> Dict[str, Any]:
    """Mock LLM response for Stage 1 (NL to LTL)"""
    return {
        "objects": ["a", "b"],
        "initial_state": [
            {"ontable": ["a"]},
            {"ontable": ["b"]},
            {"clear": ["a"]},
            {"clear": ["b"]},
            {"handempty": []}
        ],
        "ltl_formulas": [
            {
                "type": "temporal",
                "operator": "F",
                "formula": {"on": ["a", "b"]}
            },
            {
                "type": "temporal",
                "operator": "F",
                "formula": {"clear": ["a"]}
            }
        ]
    }


@pytest.fixture
def mock_llm_pddl_response() -> str:
    """Mock LLM response for Stage 2 (LTL to PDDL)"""
    return """(define (problem test_problem)
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
    (clear a)
  ))
)"""


@pytest.fixture
def mock_llm_plan_response() -> str:
    """Mock LLM response for Stage 3 (PDDL to Plan)"""
    return """[
  {"action": "pick-up", "parameters": ["a"]},
  {"action": "stack", "parameters": ["a", "b"]}
]"""


@pytest.fixture
def mock_openai_client(mock_llm_ltl_response, mock_llm_pddl_response, mock_llm_plan_response):
    """Mock OpenAI client that returns predefined responses"""
    client = MagicMock()

    # Mock chat completions
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    # Default to LTL response, can be overridden in tests
    mock_message.content = str(mock_llm_ltl_response)
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    client.chat.completions.create.return_value = mock_response

    return client


# ===== Sample Test Data Fixtures =====

@pytest.fixture
def sample_nl_instruction() -> str:
    """Sample natural language instruction"""
    return "Put block A on block B"


@pytest.fixture
def sample_ltl_spec_dict() -> Dict[str, Any]:
    """Sample LTL specification as dictionary"""
    return {
        "objects": ["a", "b"],
        "initial_state": [
            {"ontable": ["a"]},
            {"ontable": ["b"]},
            {"clear": ["a"]},
            {"clear": ["b"]},
            {"handempty": []}
        ],
        "formulas": [
            {
                "operator": "F",
                "predicate": None,
                "logical_op": None,
                "sub_formulas": [
                    {
                        "operator": None,
                        "predicate": {"on": ["a", "b"]},
                        "logical_op": None,
                        "sub_formulas": []
                    }
                ]
            }
        ],
        "formulas_string": ["F(on(a, b))"]
    }


@pytest.fixture
def sample_pddl_problem() -> str:
    """Sample PDDL problem"""
    return """(define (problem test_problem)
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


@pytest.fixture
def sample_plan() -> list:
    """Sample action plan"""
    return [
        ("pick-up", ["a"]),
        ("stack", ["a", "b"])
    ]


# ===== Temporary Directory Fixtures =====

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory for tests"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def temp_log_dir(tmp_path) -> Path:
    """Create temporary log directory for tests"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


# ===== Test Markers and Skip Conditions =====

def pytest_configure(config):
    """Configure custom markers and settings"""
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring valid OpenAI API key"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test location
        if "stage1" in str(item.fspath):
            item.add_marker(pytest.mark.stage1)
        elif "stage2" in str(item.fspath):
            item.add_marker(pytest.mark.stage2)
        elif "stage3" in str(item.fspath):
            item.add_marker(pytest.mark.stage3)

        # Mark tests that use mock_openai_client as unit tests
        if "mock_openai_client" in item.fixturenames:
            item.add_marker(pytest.mark.unit)
