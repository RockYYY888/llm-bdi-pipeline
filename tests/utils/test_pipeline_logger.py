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

    logger.log_stage3(
        None,
        None,
        None,
        "Failed",
        error="HTN synthesis failed during library_validation: bad task name",
        method="htn",
        model="deepseek-chat",
        llm_prompt={"system": "SYSTEM", "user": "USER"},
        llm_response='{"bad":"payload"}',
        metadata={
            "used_llm": False,
            "model": "deepseek-chat",
            "failure_stage": "library_validation",
            "failure_reason": "bad task name",
        },
    )
    logger.end_pipeline(success=False)

    log_dir = logger.current_log_dir
    assert log_dir is not None

    execution = json.loads((log_dir / "execution.json").read_text())
    execution_txt = (log_dir / "execution.txt").read_text()

    assert execution["stage3_status"] == "failed"
    assert execution["stage3_method"] == "htn"
    assert execution["stage3_used_llm"] is True
    assert execution["stage3_model"] == "deepseek-chat"
    assert execution["stage3_llm_prompt"]["system"] == "SYSTEM"
    assert execution["stage3_llm_response"] == '{"bad":"payload"}'
    assert execution["stage3_metadata"]["failure_stage"] == "library_validation"
    assert execution["stage3_metadata"]["failure_reason"] == "bad task name"

    assert "HTN GENERATION DIAGNOSTICS" in execution_txt
    assert "failure_stage: library_validation" in execution_txt
    assert "failure_reason: bad task name" in execution_txt
    assert "LLM RESPONSE (Stage 3)" in execution_txt
    assert "Error: HTN synthesis failed during library_validation: bad task name" in execution_txt
