import json
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.pipeline_logger import PipelineLogger


def test_stage3_failure_persists_diagnostics_in_logs(tmp_path):
    logger = PipelineLogger(logs_dir=str(tmp_path))
    logger.start_pipeline(
        "demo instruction",
        mode="dfa_agentspeak",
        domain_file="demo.hddl",
        output_dir=str(tmp_path),
    )

    logger.log_stage3_method_synthesis(
        None,
        "Failed",
        error="HTN synthesis failed during response_parse: invalid payload",
        model="deepseek-chat",
        llm_prompt={"system": "SYSTEM", "user": "USER"},
        llm_response='{"bad":"payload"}',
        metadata={
            "used_llm": True,
            "model": "deepseek-chat",
            "failure_stage": "response_parse",
            "failure_reason": "invalid payload",
        },
    )
    logger.end_pipeline(success=False)

    log_dir = logger.current_log_dir
    assert log_dir is not None

    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["stage3_status"] == "failed"
    assert execution["stage3_used_llm"] is True
    assert execution["stage3_model"] == "deepseek-chat"
    assert execution["stage3_llm_prompt"]["system"] == "SYSTEM"
    assert execution["stage3_llm_response"] == '{"bad":"payload"}'
    assert execution["stage3_metadata"]["failure_stage"] == "response_parse"
    assert execution["stage3_metadata"]["failure_reason"] == "invalid payload"

    assert "HTN METHOD SYNTHESIS DIAGNOSTICS" in execution_txt
    assert "failure_stage: response_parse" in execution_txt
    assert "failure_reason: invalid payload" in execution_txt
    assert "LLM RESPONSE (Stage 3)" in execution_txt
    assert "Error: HTN synthesis failed during response_parse: invalid payload" in execution_txt
