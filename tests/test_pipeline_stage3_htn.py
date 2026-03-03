"""
Canonical integration test for the full blocksworld example pipeline.

This test is the stable example runner for acceptance:
- it calls `pipeline.execute()`
- it passes through Stage 1, Stage 2, and Stage 3
- it verifies the logger captured each stage's input/output artifacts

Stage 1 is stubbed to keep the test deterministic and fast.
"""

import json
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline
from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
from utils.pipeline_logger import PipelineLogger


EXAMPLE_INSTRUCTION = "Using blocks a and b, arrange them so that a is on b."
EXAMPLE_PROMPT = {
    "system": "Stubbed Stage 1 system prompt for the canonical blocksworld example.",
    "user": EXAMPLE_INSTRUCTION,
}
EXAMPLE_RESPONSE = (
    '{"objects":["a","b"],"formula":"F(on(a, b))","note":"stubbed for deterministic integration test"}'
)


def _build_stage1_spec() -> LTLSpecification:
    spec = LTLSpecification()
    spec.objects = ["a", "b"]
    spec.grounding_map = GroundingMap()
    spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    atom = LTLFormula(
        operator=None,
        predicate={"on": ["a", "b"]},
        sub_formulas=[],
        logical_op=None,
    )
    spec.formulas = [
        LTLFormula(
            operator=TemporalOperator.FINALLY,
            predicate=None,
            sub_formulas=[atom],
            logical_op=None,
        )
    ]
    return spec


def test_blocksworld_example_pipeline_records_each_stage(tmp_path, monkeypatch):
    def fake_generate(self, nl_instruction):
        assert nl_instruction == EXAMPLE_INSTRUCTION
        return _build_stage1_spec(), EXAMPLE_PROMPT, EXAMPLE_RESPONSE

    monkeypatch.setattr(NLToLTLfGenerator, "generate", fake_generate)

    pipeline = LTL_BDI_Pipeline()
    pipeline.logger = PipelineLogger(logs_dir=str(tmp_path))

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

    assert execution["stage1_status"] == "success"
    assert execution["stage1_llm_prompt"] == EXAMPLE_PROMPT
    assert execution["stage1_llm_response"] == EXAMPLE_RESPONSE
    assert execution["stage1_ltlf_spec"]["formulas_string"] == ["F(on(a, b))"]
    assert execution["stage1_ltlf_spec"]["objects"] == ["a", "b"]

    assert execution["stage2_status"] == "success"
    assert execution["stage2_formula"] == "F(on(a, b))"
    assert execution["stage2_num_states"] == 2
    assert execution["stage2_num_transitions"] == 4

    assert execution["stage3_status"] == "success"
    assert execution["stage3_method"] == "htn"
    assert execution["stage3_code_size_chars"] > 0
    assert execution["stage3_metadata"]["method"] == "htn"
    assert execution["stage3_metadata"]["transition_count"] >= 1
    assert execution["stage3_artifacts"]["method_library"] is not None
    assert execution["stage3_artifacts"]["transitions"] is not None

    assert "STAGE 1: Natural Language" in execution_txt
    assert "PARSED OUTPUT (Stage 1)" in execution_txt
    assert "STAGE 2: LTL Specification" in execution_txt
    assert "DFA GENERATION RESULT" in execution_txt
    assert "STAGE 3: DFA" in execution_txt
    assert "HTN GENERATION SUMMARY" in execution_txt
    assert "HTN METHOD LIBRARY (Stage 3A)" in execution_txt
    assert "HTN DECOMPOSITION TRACES (Stage 3B/3C)" in execution_txt
    assert "GENERATED AGENTSPEAK CODE (Stage 3)" in execution_txt
