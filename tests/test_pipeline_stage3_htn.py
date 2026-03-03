"""
Pipeline-level integration test for the new HTN Stage 3 path.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ltl_bdi_pipeline import LTL_BDI_Pipeline
from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator
from stage2_dfa_generation.dfa_builder import DFABuilder


def test_pipeline_stage3_writes_htn_artifacts(tmp_path):
    pipeline = LTL_BDI_Pipeline()
    pipeline.output_dir = tmp_path

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

    dfa_result = DFABuilder().build(spec)
    code, stats = pipeline._stage3_htn_generation(spec, dfa_result)

    assert stats["method"] == "htn"
    assert "+!transition_1" in code
    assert (tmp_path / "agentspeak_generated.asl").exists()
    assert (tmp_path / "htn_method_library.json").exists()
    assert (tmp_path / "htn_transitions.json").exists()
