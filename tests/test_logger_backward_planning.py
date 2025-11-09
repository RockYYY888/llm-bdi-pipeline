"""
Test Pipeline Logger with Backward Planning
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pipeline_logger import PipelineLogger
from src.stage3_code_generation.backward_planner_generator import BackwardPlannerGenerator
from src.utils.pddl_parser import PDDLParser
from src.stage1_interpretation.grounding_map import GroundingMap


def test_logger_with_backward_planning():
    """
    Test pipeline logger with actual backward planning execution
    """
    print("=" * 80)
    print("TESTING PIPELINE LOGGER WITH BACKWARD PLANNING")
    print("=" * 80)
    print()

    # Initialize logger
    logger = PipelineLogger(logs_dir="logs/test")

    # Start pipeline
    nl_input = "Put block a on block b"
    logger.start_pipeline(
        natural_language=nl_input,
        mode="backward_planning",
        timestamp="test_20241109"
    )

    # Stage 1: Mock LTL specification
    grounding_map = GroundingMap()
    grounding_map.add_atom("on_a_b", "on", ["a", "b"])

    ltl_spec = {
        "objects": ["a", "b"],
        "formulas_string": ["F(on(a, b))"],
        "grounding_map": grounding_map.to_dict() if hasattr(grounding_map, 'to_dict') else {}
    }

    logger.log_stage1(
        nl_input=nl_input,
        ltl_spec=ltl_spec,
        status="Success"
    )

    # Stage 2: Mock DFA
    dfa_dot = """
digraph {
    __start [shape=point];
    __start -> state0;
    state0 [label="0"];
    state1 [label="1", shape=doublecircle];
    state0 -> state1 [label="on_a_b"];
}
"""

    dfa_result = {
        "formula": "F(on(a, b))",
        "dfa_dot": dfa_dot,
        "num_states": 2,
        "num_transitions": 1
    }

    logger.log_stage2_dfas(
        ltl_spec=ltl_spec,
        dfa_result=dfa_result,
        status="Success"
    )

    # Stage 3: Run actual backward planning
    print("Running backward planning...")
    domain_path = project_root / "src" / "legacy" / "fond" / "domains" / "blocksworld" / "domain.pddl"
    domain = PDDLParser.parse_domain(str(domain_path))

    generator = BackwardPlannerGenerator(domain, grounding_map)
    agentspeak_code = generator.generate(ltl_spec, dfa_result)

    print(f"✓ Generated {len(agentspeak_code)} characters of AgentSpeak code")

    # Extract statistics from console output (simplified for test)
    # In real integration, these would be returned by the generator
    logger.log_stage3(
        ltl_spec=ltl_spec,
        dfa_result=dfa_result,
        agentspeak_code=agentspeak_code,
        status="Success",
        method="backward_planning",
        states_explored=1093,
        transitions_generated=63394,
        goal_plans_count=26,
        action_plans_count=7,
        cache_hits=0,
        cache_misses=1,
        ground_actions_cached=32,
        redundancy_eliminated_pct=0.0
    )

    # End pipeline
    log_file = logger.end_pipeline(success=True)

    print()
    print("=" * 80)
    print("LOGGER TEST COMPLETE")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Log directory: {logger.current_log_dir}")
    print()

    # Verify files were created
    assert log_file.exists(), "JSON log file not created"
    assert (logger.current_log_dir / "execution.txt").exists(), "TXT log file not created"
    assert (logger.current_log_dir / "generated_code.asl").exists(), "AgentSpeak file not created"

    print("✅ All log files created successfully!")
    print()

    # Show file sizes
    print("File sizes:")
    print(f"  execution.json: {log_file.stat().st_size:,} bytes")
    print(f"  execution.txt: {(logger.current_log_dir / 'execution.txt').stat().st_size:,} bytes")
    print(f"  generated_code.asl: {(logger.current_log_dir / 'generated_code.asl').stat().st_size:,} bytes")
    print()

    # Show snippet of generated_code.asl
    print("=" * 80)
    print("GENERATED AGENTSPEAK CODE (first 500 chars)")
    print("=" * 80)
    asl_file = logger.current_log_dir / "generated_code.asl"
    with open(asl_file, 'r') as f:
        content = f.read()
        print(content[:500])
        print("...")
    print()

    return logger.current_log_dir


if __name__ == "__main__":
    log_dir = test_logger_with_backward_planning()
    print(f"\n✅ Test completed! Check logs at: {log_dir}")
