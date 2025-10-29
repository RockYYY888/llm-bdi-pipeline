"""
Pipeline Logger

Records the complete execution trace of the LTL pipeline:
- Natural language input
- LTL specification generated
- PDDL problem created
- Plan solution
- Any errors encountered

Each run is saved with a timestamp for easy tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PipelineRecord:
    """Complete record of a pipeline execution"""
    timestamp: str
    natural_language: str
    success: bool
    mode: str = "both"  # "both", "llm_agentspeak", or "fond"

    # Stage 1: NL -> LTL
    stage1_status: str  # "success" | "failed" | "skipped"
    stage1_ltl_spec: Optional[Dict[str, Any]] = None
    stage1_error: Optional[str] = None
    stage1_used_llm: bool = False
    stage1_model: Optional[str] = None
    stage1_llm_prompt: Optional[Dict[str, str]] = None  # {"system": "...", "user": "..."}
    stage1_llm_response: Optional[str] = None

    # Stage 2: LTL -> PDDL
    stage2_status: str = "skipped"
    stage2_pddl: Optional[str] = None
    stage2_error: Optional[str] = None
    stage2_used_llm: bool = False
    stage2_model: Optional[str] = None
    stage2_llm_prompt: Optional[Dict[str, str]] = None  # {"system": "...", "user": "..."}
    stage2_llm_response: Optional[str] = None

    # Stage 3: PDDL -> Plan
    stage3_status: str = "skipped"
    stage3_plan: Optional[List[Tuple[str, List[str]]]] = None
    stage3_error: Optional[str] = None
    stage3_used_llm: bool = False
    stage3_model: Optional[str] = None
    stage3_llm_prompt: Optional[Dict[str, str]] = None  # {"system": "...", "user": "..."}
    stage3_llm_response: Optional[str] = None

    # Metadata
    domain_file: str = "domains/blocksworld/domain.pddl"
    output_dir: str = "output"
    execution_time_seconds: float = 0.0


class PipelineLogger:
    """
    Logger for LTL pipeline executions

    Saves structured JSON records with timestamps for each pipeline run.
    """

    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize logger

        Args:
            logs_dir: Directory to save log files
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.current_record: Optional[PipelineRecord] = None
        self.start_time: Optional[datetime] = None
        self.current_log_dir: Optional[Path] = None

    def start_pipeline(self,
                      natural_language: str,
                      mode: str = "both",
                      domain_file: str = "domains/blocksworld/domain.pddl",
                      output_dir: str = "output",
                      timestamp: str = None):
        """
        Start logging a new pipeline execution

        Args:
            natural_language: The input instruction
            mode: Execution mode - "both", "llm_agentspeak", or "fond"
            domain_file: PDDL domain file used
            output_dir: Output directory for generated files
            timestamp: Optional timestamp string (YYYYMMDD_HHMMSS format). If not provided, current time is used.
        """
        self.start_time = datetime.now()
        if timestamp is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Create timestamp-specific log directory
        self.current_log_dir = self.logs_dir / timestamp
        self.current_log_dir.mkdir(parents=True, exist_ok=True)

        self.current_record = PipelineRecord(
            timestamp=timestamp,
            natural_language=natural_language,
            success=False,
            mode=mode,
            stage1_status="pending",
            stage2_status="pending",
            stage3_status="pending",
            domain_file=domain_file,
            output_dir=output_dir
        )

    def log_stage1_success(self,
                          ltl_spec: Dict[str, Any],
                          used_llm: bool = False,
                          model: Optional[str] = None,
                          llm_prompt: Optional[Dict[str, str]] = None,
                          llm_response: Optional[str] = None):
        """
        Log successful Stage 1 completion

        Args:
            ltl_spec: The generated LTL specification
            used_llm: Whether LLM was used (vs mock parser)
            model: Model name if LLM was used
            llm_prompt: LLM prompt (system + user messages)
            llm_response: Raw LLM response text
        """
        if not self.current_record:
            return

        self.current_record.stage1_status = "success"
        self.current_record.stage1_ltl_spec = ltl_spec
        self.current_record.stage1_used_llm = used_llm
        self.current_record.stage1_model = model
        self.current_record.stage1_llm_prompt = llm_prompt
        self.current_record.stage1_llm_response = llm_response

    def log_stage1_error(self, error: str):
        """Log Stage 1 failure"""
        if not self.current_record:
            return

        self.current_record.stage1_status = "failed"
        self.current_record.stage1_error = str(error)

    def log_stage2_success(self,
                          pddl_problem: str,
                          used_llm: bool = False,
                          model: Optional[str] = None,
                          llm_prompt: Optional[Dict[str, str]] = None,
                          llm_response: Optional[str] = None):
        """
        Log successful Stage 2 completion

        Args:
            pddl_problem: The generated PDDL problem
            used_llm: Whether LLM was used (vs template fallback)
            model: Model name if LLM was used
            llm_prompt: LLM prompt (system + user messages)
            llm_response: Raw LLM response text
        """
        if not self.current_record:
            return

        self.current_record.stage2_status = "success"
        self.current_record.stage2_pddl = pddl_problem
        self.current_record.stage2_used_llm = used_llm
        self.current_record.stage2_model = model
        self.current_record.stage2_llm_prompt = llm_prompt
        self.current_record.stage2_llm_response = llm_response

    def log_stage2_error(self, error: str):
        """Log Stage 2 failure"""
        if not self.current_record:
            return

        self.current_record.stage2_status = "failed"
        self.current_record.stage2_error = str(error)

    def log_stage3_success(self,
                          plan: List[Tuple[str, List[str]]],
                          used_llm: bool = False,
                          model: Optional[str] = None,
                          prompt: Optional[Dict[str, str]] = None,
                          response: Optional[str] = None):
        """
        Log successful Stage 3 completion

        Args:
            plan: The generated plan as list of (action, params) tuples
            used_llm: Whether LLM planner was used (vs classical planner)
            model: Model name if LLM was used
            prompt: LLM prompt (system + user messages)
            response: Raw LLM response text
        """
        if not self.current_record:
            return

        self.current_record.stage3_status = "success"
        self.current_record.stage3_plan = plan
        self.current_record.stage3_used_llm = used_llm
        self.current_record.stage3_model = model
        self.current_record.stage3_llm_prompt = prompt
        self.current_record.stage3_llm_response = response

    def log_stage3_error(self, error: str):
        """Log Stage 3 failure"""
        if not self.current_record:
            return

        self.current_record.stage3_status = "failed"
        self.current_record.stage3_error = str(error)

    # Simplified logging helpers for dual-branch pipeline
    def log_stage1(self, nl_input: str, ltl_spec: Any, status: str, error: str = None):
        """Simplified Stage 1 logger"""
        if status == "Success" and ltl_spec:
            self.log_stage1_success(ltl_spec.to_dict() if hasattr(ltl_spec, 'to_dict') else ltl_spec, used_llm=True)
        elif error:
            self.log_stage1_error(error)

    def log_stage2(self, ltl_spec: Any, pddl_problem: Any, status: str, error: str = None):
        """Simplified Stage 2 logger"""
        if status == "Success" and pddl_problem:
            self.log_stage2_success(str(pddl_problem), used_llm=True)
        elif error:
            self.log_stage2_error(error)

    def log_stage3a(self, pddl_problem: Any, plan: List[Tuple[str, List[str]]], status: str, error: str = None):
        """Simplified Stage 3A (Classical) logger"""
        if status == "Success" and plan:
            self.log_stage3_success(plan, used_llm=False)
        elif error:
            self.log_stage3_error(error)

    def log_stage3b(self, ltl_spec: Any, asl_code: str, status: str, error: str = None):
        """Simplified Stage 3B (AgentSpeak) logger - stores in unused fields"""
        # For MVP: Store in llm_response field since we don't have dedicated stage3b fields
        pass

    def log_stage4(self, results: Dict[str, Any], status: str, error: str = None):
        """Simplified Stage 4 (Execution) logger"""
        # For MVP: Just track success/failure
        pass

    def end_pipeline(self, success: bool = True) -> Path:
        """
        End logging and save the record

        Args:
            success: Whether the overall pipeline succeeded

        Returns:
            Path to the saved log file
        """
        if not self.current_record or not self.start_time:
            raise RuntimeError("No active pipeline record to end")

        # Calculate execution time
        end_time = datetime.now()
        self.current_record.execution_time_seconds = (
            end_time - self.start_time
        ).total_seconds()

        # Set overall success status
        self.current_record.success = success

        # Save to timestamp directory
        if not self.current_log_dir:
            raise RuntimeError("Log directory not initialized")

        # Save JSON log
        json_filepath = self.current_log_dir / "execution.json"
        record_dict = asdict(self.current_record)

        with open(json_filepath, 'w') as f:
            json.dump(record_dict, f, indent=2)

        # Save human-readable format
        txt_filepath = self.current_log_dir / "execution.txt"
        self._save_readable_format(txt_filepath, record_dict)

        return json_filepath

    def _save_readable_format(self, filepath: Path, record: Dict[str, Any]):
        """Save a human-readable text version of the record"""
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LTL PIPELINE EXECUTION RECORD\n")
            f.write("="*80 + "\n\n")

            # Header
            f.write(f"Timestamp: {record['timestamp']}\n")
            f.write(f"Execution Time: {record['execution_time_seconds']:.2f} seconds\n")
            f.write(f"Overall Status: {'✓ SUCCESS' if record['success'] else '✗ FAILED'}\n")
            f.write(f"Domain: {record['domain_file']}\n")
            f.write(f"Output Directory: {record['output_dir']}\n")
            f.write("\n")

            # Input
            f.write("-"*80 + "\n")
            f.write("INPUT\n")
            f.write("-"*80 + "\n")
            f.write(f"Natural Language: \"{record['natural_language']}\"\n")
            f.write("\n")

            # Stage 1
            f.write("-"*80 + "\n")
            f.write("STAGE 1: Natural Language → LTL Specification\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record['stage1_status'].upper()}\n")

            if record['stage1_used_llm']:
                f.write(f"Parser: LLM ({record['stage1_model']})\n")
            else:
                f.write("Parser: Mock\n")

            # LLM Prompt/Response for Stage 1
            if record['stage1_used_llm'] and record['stage1_llm_prompt']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM PROMPT (Stage 1)\n")
                f.write("~"*40 + "\n")
                prompt = record['stage1_llm_prompt']
                f.write("\n[SYSTEM MESSAGE]\n")
                f.write(prompt.get('system', 'N/A')[:500])  # First 500 chars
                if len(prompt.get('system', '')) > 500:
                    f.write(f"\n... (truncated, total {len(prompt.get('system', ''))} chars)")
                f.write("\n\n[USER MESSAGE]\n")
                f.write(prompt.get('user', 'N/A'))
                f.write("\n")

            if record['stage1_used_llm'] and record['stage1_llm_response']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM RESPONSE (Stage 1)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage1_llm_response'][:1000])  # First 1000 chars
                if len(record['stage1_llm_response']) > 1000:
                    f.write(f"\n... (truncated, total {len(record['stage1_llm_response'])} chars)")
                f.write("\n")

            if record['stage1_status'] == 'success' and record['stage1_ltl_spec']:
                f.write("\n" + "~"*40 + "\n")
                f.write("PARSED OUTPUT (Stage 1)\n")
                f.write("~"*40 + "\n")
                ltl = record['stage1_ltl_spec']
                f.write(f"Objects: {ltl.get('objects', [])}\n")
                f.write(f"\nInitial State ({len(ltl.get('initial_state', []))} predicates):\n")
                for pred in ltl.get('initial_state', []):
                    for name, args in pred.items():
                        if args:
                            f.write(f"  - {name}({', '.join(args)})\n")
                        else:
                            f.write(f"  - {name}\n")

                f.write(f"\nLTL Formulas:\n")
                for i, formula_str in enumerate(ltl.get('formulas_string', []), 1):
                    f.write(f"  {i}. {formula_str}\n")

            elif record['stage1_error']:
                f.write(f"\nError: {record['stage1_error']}\n")

            f.write("\n")

            # Stage 2
            f.write("-"*80 + "\n")
            f.write("STAGE 2: LTL Specification → PDDL Problem\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record['stage2_status'].upper()}\n")

            if record['stage2_used_llm']:
                f.write(f"Converter: LLM ({record['stage2_model']})\n")
            else:
                f.write("Converter: Template Fallback\n")

            # LLM Prompt/Response for Stage 2
            if record['stage2_used_llm'] and record['stage2_llm_prompt']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM PROMPT (Stage 2)\n")
                f.write("~"*40 + "\n")
                prompt = record['stage2_llm_prompt']
                f.write("\n[SYSTEM MESSAGE]\n")
                f.write(prompt.get('system', 'N/A')[:500])  # First 500 chars
                if len(prompt.get('system', '')) > 500:
                    f.write(f"\n... (truncated, total {len(prompt.get('system', ''))} chars)")
                f.write("\n\n[USER MESSAGE]\n")
                f.write(prompt.get('user', 'N/A')[:500])  # First 500 chars
                if len(prompt.get('user', '')) > 500:
                    f.write(f"\n... (truncated, total {len(prompt.get('user', ''))} chars)")
                f.write("\n")

            if record['stage2_used_llm'] and record['stage2_llm_response']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM RESPONSE (Stage 2)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage2_llm_response'][:1000])  # First 1000 chars
                if len(record['stage2_llm_response']) > 1000:
                    f.write(f"\n... (truncated, total {len(record['stage2_llm_response'])} chars)")
                f.write("\n")

            if record['stage2_status'] == 'success' and record['stage2_pddl']:
                f.write("\n" + "~"*40 + "\n")
                f.write("GENERATED PDDL PROBLEM (Stage 2)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage2_pddl'])
                f.write("\n")

            elif record['stage2_error']:
                f.write(f"\nError: {record['stage2_error']}\n")

            f.write("\n")

            # Stage 3
            f.write("-"*80 + "\n")
            f.write("STAGE 3: PDDL Problem → Plan\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record['stage3_status'].upper()}\n")

            if record['stage3_used_llm']:
                f.write(f"Planner: LLM ({record['stage3_model']})\n")
            else:
                f.write("Planner: Classical PDDL Planner (pyperplan)\n")

            # LLM Prompt/Response for Stage 3
            if record['stage3_used_llm'] and record['stage3_llm_prompt']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM PROMPT (Stage 3)\n")
                f.write("~"*40 + "\n")
                prompt = record['stage3_llm_prompt']
                f.write("\n[SYSTEM MESSAGE]\n")
                f.write(prompt.get('system', 'N/A')[:500])  # First 500 chars
                if len(prompt.get('system', '')) > 500:
                    f.write(f"\n... (truncated, total {len(prompt.get('system', ''))} chars)")
                f.write("\n\n[USER MESSAGE]\n")
                f.write(prompt.get('user', 'N/A'))
                f.write("\n")

            if record['stage3_used_llm'] and record['stage3_llm_response']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM RESPONSE (Stage 3)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage3_llm_response'][:1000])  # First 1000 chars
                if len(record['stage3_llm_response']) > 1000:
                    f.write(f"\n... (truncated, total {len(record['stage3_llm_response'])} chars)")
                f.write("\n")

            if record['stage3_status'] == 'success' and record['stage3_plan']:
                f.write("\n" + "~"*40 + "\n")
                f.write("GENERATED PLAN (Stage 3)\n")
                f.write("~"*40 + "\n")
                f.write(f"Plan ({len(record['stage3_plan'])} actions):\n")
                for i, (action, params) in enumerate(record['stage3_plan'], 1):
                    f.write(f"  {i}. {action}({', '.join(params)})\n")

            elif record['stage3_error']:
                f.write(f"\nError: {record['stage3_error']}\n")

            f.write("\n")

            # Footer
            f.write("="*80 + "\n")
            f.write("END OF RECORD\n")
            f.write("="*80 + "\n")


def test_logger():
    """Test the pipeline logger"""
    logger = PipelineLogger()

    # Start pipeline
    logger.start_pipeline("Put block A on block B")

    # Log Stage 1
    logger.log_stage1_success(
        ltl_spec={
            "objects": ["a", "b"],
            "initial_state": [
                {"ontable": ["a"]},
                {"ontable": ["b"]},
                {"clear": ["a"]},
                {"clear": ["b"]},
                {"handempty": []}
            ],
            "formulas_string": ["F(on(a, b))", "F(clear(a))"]
        },
        used_llm=True,
        model="deepseek-chat"
    )

    # Log Stage 2
    pddl = """(define (problem test)
  (:domain blocksworld)
  (:objects a b)
  (:init (ontable a) (ontable b) (clear a) (clear b) (handempty))
  (:goal (and (on a b)))
)"""
    logger.log_stage2_success(pddl)

    # Log Stage 3
    logger.log_stage3_success([
        ("pickup", ["a"]),
        ("stack", ["a", "b"])
    ])

    # End pipeline
    filepath = logger.end_pipeline(success=True)
    print(f"Log saved to: {filepath}")


if __name__ == "__main__":
    test_logger()
