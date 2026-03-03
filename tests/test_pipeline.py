"""
Live integration test for the full blocksworld example pipeline.

This test mirrors `src/main.py` as closely as possible:
- it fixes only the user instruction
- it passes through the real Stage 1, Stage 2, and Stage 3 execution path
- it verifies the logger captured the actual prompts, responses, and artifacts

The test is skipped when a valid API configuration is not available.
"""

import json
import sys
from pathlib import Path

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline
from utils.config import get_config
from utils.pipeline_logger import PipelineLogger


EXAMPLE_INSTRUCTION = "Using blocks a and b, arrange them so that a is on b."


def test_blocksworld_example_pipeline_records_live_execution():
    config = get_config()
    if not config.validate():
        pytest.skip(
            "Live pipeline test requires a valid OPENAI_API_KEY, matching src/main.py",
        )

    pipeline = LTL_BDI_Pipeline()
    test_logs_dir = Path(__file__).parent / "logs"
    pipeline.logger = PipelineLogger(logs_dir=str(test_logs_dir))

    result = pipeline.execute(EXAMPLE_INSTRUCTION, mode="dfa_agentspeak")

    assert result["success"] is True
    assert "agentspeak_code" in result

    log_dir = pipeline.logger.current_log_dir
    assert log_dir is not None

    execution_json_path = log_dir / "execution.json"
    execution_txt_path = log_dir / "execution.txt"
    grounding_map_path = log_dir / "grounding_map.json"
    dfa_original_path = log_dir / "dfa_original.dot"
    dfa_simplified_path = log_dir / "dfa_simplified.dot"
    generated_code_path = log_dir / "generated_code.asl"
    agentspeak_path = log_dir / "agentspeak_generated.asl"
    method_library_path = log_dir / "htn_method_library.json"
    transitions_path = log_dir / "htn_transitions.json"
    dfa_json_path = log_dir / "dfa.json"

    for path in [
        execution_json_path,
        execution_txt_path,
        grounding_map_path,
        dfa_original_path,
        dfa_simplified_path,
        generated_code_path,
        agentspeak_path,
        method_library_path,
        transitions_path,
        dfa_json_path,
    ]:
        assert path.exists(), f"missing expected log artifact: {path.name}"

    execution = json.loads(execution_json_path.read_text())
    execution_txt = execution_txt_path.read_text()

    assert execution["natural_language"] == EXAMPLE_INSTRUCTION
    assert execution["success"] is True

    assert execution["stage1_status"] == "success"
    assert execution["stage1_used_llm"] is True
    assert execution["stage1_model"] == config.openai_model
    assert execution["stage1_llm_prompt"] is not None
    assert execution["stage1_llm_prompt"]["system"]
    assert EXAMPLE_INSTRUCTION in execution["stage1_llm_prompt"]["user"]
    assert execution["stage1_llm_response"]
    assert set(execution["stage1_ltlf_spec"]["objects"]) == {"a", "b"}
    assert execution["stage1_ltlf_spec"]["formulas_string"]

    assert execution["stage2_status"] == "success"
    assert execution["stage2_formula"]
    assert execution["stage2_num_states"] >= 1
    assert execution["stage2_num_transitions"] >= 1

    assert execution["stage3_status"] == "success"
    assert execution["stage3_method"] == "htn"
    assert execution["stage3_used_llm"] is True
    assert execution["stage3_llm_prompt"] is not None
    assert execution["stage3_metadata"]["llm_attempted"] is True
    assert execution["stage3_code_size_chars"] > 0
    assert execution["stage3_code_size_chars"] == len(execution["stage3_agentspeak"])
    assert execution["stage3_metadata"]["method"] == "htn"
    assert execution["stage3_metadata"]["transition_count"] >= 1
    assert execution["stage3_artifacts"]["method_library"] is not None
    assert execution["stage3_artifacts"]["transitions"] is not None
    assert result["agentspeak_code"] == execution["stage3_agentspeak"]

    assert "STAGE 1: Natural Language" in execution_txt
    assert "Parser: LLM (" in execution_txt
    assert "Parser: Mock" not in execution_txt
    assert "LLM RESPONSE (Stage 1)" in execution_txt
    assert "PARSED OUTPUT (Stage 1)" in execution_txt
    assert "STAGE 2: LTL Specification" in execution_txt
    assert "DFA GENERATION RESULT" in execution_txt
    assert "STAGE 3: DFA" in execution_txt
    assert "HTN GENERATION SUMMARY" in execution_txt
    assert "HTN METHOD LIBRARY (Stage 3A)" in execution_txt
    assert "HTN DECOMPOSITION TRACES (Stage 3B/3C)" in execution_txt
    assert "GENERATED AGENTSPEAK CODE (Stage 3)" in execution_txt
