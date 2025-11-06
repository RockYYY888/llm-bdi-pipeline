"""
Pipeline Logger

Records the complete execution trace of the LTLf-BDI pipeline:
- Natural language input
- LTLf specification generated
- DFA conversion (optional)
- AgentSpeak code generation
- Any errors encountered

Each run is saved with a timestamp for easy tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PipelineRecord:
    """Complete record of a pipeline execution"""
    timestamp: str
    natural_language: str
    success: bool
    mode: str = "dfa_agentspeak"

    # Stage 1: NL -> LTLf
    stage1_status: str = "pending"
    stage1_ltlf_spec: Optional[Dict[str, Any]] = None
    stage1_error: Optional[str] = None
    stage1_used_llm: bool = False
    stage1_model: Optional[str] = None
    stage1_llm_prompt: Optional[Dict[str, str]] = None
    stage1_llm_response: Optional[str] = None

    # Stage 2: LTLf -> DFA Generation
    stage2_status: str = "pending"
    stage2_dfa_result: Optional[Dict[str, Any]] = None  # DFA dict (formula, dfa_dot, num_states, num_transitions)
    stage2_formula: Optional[str] = None
    stage2_num_states: int = 0
    stage2_num_transitions: int = 0
    stage2_error: Optional[str] = None

    # Stage 3: DFAs -> AgentSpeak Code Generation
    stage3_status: str = "pending"
    stage3_agentspeak: Optional[str] = None
    stage3_error: Optional[str] = None
    stage3_used_llm: bool = False
    stage3_model: Optional[str] = None
    stage3_llm_prompt: Optional[Dict[str, str]] = None
    stage3_llm_response: Optional[str] = None

    # Metadata
    domain_file: str = "domains/blocksworld/domain.pddl"
    output_dir: str = "output"
    execution_time_seconds: float = 0.0


class PipelineLogger:
    """
    Logger for LTLf-BDI pipeline executions

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

        # Create timestamp and mode-specific log directory
        # Format: YYYYMMDD_HHMMSS_mode (e.g., 20251029_222949_fond)
        dir_name = f"{timestamp}_{mode}"
        self.current_log_dir = self.logs_dir / dir_name
        self.current_log_dir.mkdir(parents=True, exist_ok=True)

        self.current_record = PipelineRecord(
            timestamp=timestamp,
            natural_language=natural_language,
            success=False,
            mode=mode,
            stage1_status="pending",
            stage2_status="pending",
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
        self.current_record.stage1_ltlf_spec = ltl_spec
        self.current_record.stage1_used_llm = used_llm
        self.current_record.stage1_model = model
        self.current_record.stage1_llm_prompt = llm_prompt
        self.current_record.stage1_llm_response = llm_response

        # Save grounding_map.json if present
        if ltl_spec and 'grounding_map' in ltl_spec and self.current_log_dir:
            import json
            grounding_map_path = self.current_log_dir / "grounding_map.json"
            with open(grounding_map_path, 'w') as f:
                json.dump(ltl_spec['grounding_map'], f, indent=2)

        # IMMEDIATELY save current state to files
        self._save_current_state()

    def log_stage1_error(self, error: str):
        """Log Stage 1 failure"""
        if not self.current_record:
            return

        self.current_record.stage1_status = "failed"
        self.current_record.stage1_error = str(error)

        # IMMEDIATELY save current state to files
        self._save_current_state()

    # NEW: Stage 2 - DFA Generation logging
    def log_stage2_dfas(self, ltl_spec: Any, dfa_result: Any, status: str, error: str = None):
        """Log Stage 2: LTLf -> DFA Generation"""
        if not self.current_record:
            return

        if status == "Success" and dfa_result:
            self.current_record.stage2_status = "success"
            self.current_record.stage2_dfa_result = dfa_result
            self.current_record.stage2_formula = dfa_result.get('formula', 'N/A')
            self.current_record.stage2_num_states = dfa_result.get('num_states', 0)
            self.current_record.stage2_num_transitions = dfa_result.get('num_transitions', 0)
        elif error:
            self.current_record.stage2_status = "failed"
            self.current_record.stage2_error = str(error)

        # IMMEDIATELY save current state to files
        self._save_current_state()

    # NEW: Stage 3 - AgentSpeak Generation logging
    def log_stage3(self, ltl_spec: Any, dfa_result: Any, agentspeak_code: str, status: str,
                   error: str = None, model: str = None, llm_prompt: Dict[str, str] = None,
                   llm_response: str = None):
        """Log Stage 3: DFAs -> AgentSpeak Code Generation"""
        if not self.current_record:
            return

        # ALWAYS log model and prompt if provided (regardless of success/failure)
        if model:
            self.current_record.stage3_used_llm = True
            self.current_record.stage3_model = model
        if llm_prompt:
            self.current_record.stage3_llm_prompt = llm_prompt
        if llm_response:
            self.current_record.stage3_llm_response = llm_response

        if status == "Success" and agentspeak_code:
            self.current_record.stage3_status = "success"
            self.current_record.stage3_agentspeak = agentspeak_code
        elif error:
            self.current_record.stage3_status = "failed"
            self.current_record.stage3_error = str(error)

        # IMMEDIATELY save current state to files
        self._save_current_state()

    # Simplified logging helper for Stage 1
    def log_stage1(self, nl_input: str, ltl_spec: Any, status: str, error: str = None,
                   model: str = None, llm_prompt: Dict[str, str] = None, llm_response: str = None):
        """Simplified Stage 1 logger"""
        if status == "Success" and ltl_spec:
            self.log_stage1_success(
                ltl_spec.to_dict() if hasattr(ltl_spec, 'to_dict') else ltl_spec,
                used_llm=True,
                model=model,
                llm_prompt=llm_prompt,
                llm_response=llm_response
            )
        elif error:
            self.log_stage1_error(error)


    def _save_current_state(self):
        """
        IMMEDIATELY save current state to both JSON and TXT files
        Called after each stage completion (success or failure)
        """
        if not self.current_record or not self.current_log_dir:
            return

        # Calculate current execution time
        if self.start_time:
            current_time = datetime.now()
            self.current_record.execution_time_seconds = (
                current_time - self.start_time
            ).total_seconds()

        # Save JSON log
        json_filepath = self.current_log_dir / "execution.json"
        record_dict = asdict(self.current_record)

        with open(json_filepath, 'w') as f:
            json.dump(record_dict, f, indent=2)

        # Save human-readable format
        txt_filepath = self.current_log_dir / "execution.txt"
        self._save_readable_format(txt_filepath, record_dict)

    def end_pipeline(self, success: bool = True) -> Path:
        """
        End logging and save the final record

        Args:
            success: Whether the overall pipeline succeeded

        Returns:
            Path to the saved log file
        """
        if not self.current_record or not self.start_time:
            raise RuntimeError("No active pipeline record to end")

        # Set overall success status
        self.current_record.success = success

        # Save final state (this will update execution time)
        self._save_current_state()

        # Return path to JSON log
        if not self.current_log_dir:
            raise RuntimeError("Log directory not initialized")

        return self.current_log_dir / "execution.json"

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
            f.write(f"\"{record['natural_language']}\"\n")
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
                f.write("System Prompt\n")
                f.write("~"*40 + "\n")
                prompt = record['stage1_llm_prompt']
                f.write(prompt.get('system', 'N/A'))  # Full system prompt
                f.write("\n\n" + "~"*40 + "\n")
                f.write("User Prompt\n")
                f.write("~"*40 + "\n")
                f.write(prompt.get('user', 'N/A'))  # Full user prompt
                f.write("\n")

            if record['stage1_used_llm'] and record['stage1_llm_response']:
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM RESPONSE (Stage 1)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage1_llm_response'])  # Full response
                f.write("\n")

            if record['stage1_status'] == 'success' and record['stage1_ltlf_spec']:
                f.write("\n" + "~"*40 + "\n")
                f.write("PARSED OUTPUT (Stage 1)\n")
                f.write("~"*40 + "\n")
                ltlf = record['stage1_ltlf_spec']
                f.write(f"Objects: {ltlf.get('objects', [])}\n")
                f.write(f"\nLTLf Formulas (goal-oriented, no initial state assumptions):\n")
                for i, formula_str in enumerate(ltlf.get('formulas_string', []), 1):
                    f.write(f"  {i}. {formula_str}\n")

            elif record['stage1_error']:
                f.write(f"\nError: {record['stage1_error']}\n")

            f.write("\n")

            # Stage 2: DFA Generation
            f.write("-"*80 + "\n")
            f.write("STAGE 2: LTL Specification → DFA Generation\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record['stage2_status'].upper()}\n")

            if record['stage2_status'] == 'success' and record.get('stage2_dfa_result'):
                f.write("\n" + "~"*40 + "\n")
                f.write("DFA GENERATION RESULT\n")
                f.write("~"*40 + "\n")

                f.write(f"Formula: {record.get('stage2_formula', 'N/A')}\n")
                f.write(f"States: {record.get('stage2_num_states', 0)}\n")
                f.write(f"Transitions: {record.get('stage2_num_transitions', 0)}\n")

                # Optionally show a snippet of the DFA DOT format
                dfa_result = record['stage2_dfa_result']
                dfa_dot = dfa_result.get('dfa_dot', '')
                if dfa_dot:
                    f.write(f"\nDFA DOT Format (first 300 chars):\n")
                    f.write(dfa_dot[:300] + "...\n")

                f.write("\n")

            elif record.get('stage2_error'):
                f.write(f"\nError: {record['stage2_error']}\n")

            f.write("\n")

            # Stage 3: AgentSpeak Generation
            f.write("-"*80 + "\n")
            f.write("STAGE 3: DFAs → AgentSpeak Code Generation\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record.get('stage3_status', 'pending').upper()}\n")

            if record.get('stage3_used_llm'):
                f.write(f"Generator: LLM ({record.get('stage3_model', 'N/A')})\n")

            # LLM Prompt/Response for Stage 3
            if record.get('stage3_used_llm') and record.get('stage3_llm_prompt'):
                f.write("\n" + "~"*40 + "\n")
                f.write("System Prompt\n")
                f.write("~"*40 + "\n")
                prompt = record['stage3_llm_prompt']
                f.write(prompt.get('system', 'N/A'))
                f.write("\n\n" + "~"*40 + "\n")
                f.write("User Prompt\n")
                f.write("~"*40 + "\n")
                f.write(prompt.get('user', 'N/A'))
                f.write("\n")

            if record.get('stage3_used_llm') and record.get('stage3_llm_response'):
                f.write("\n" + "~"*40 + "\n")
                f.write("LLM RESPONSE (Stage 3)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage3_llm_response'])
                f.write("\n")

            if record.get('stage3_status') == 'success' and record.get('stage3_agentspeak'):
                f.write("\n" + "~"*40 + "\n")
                f.write("GENERATED AGENTSPEAK CODE (Stage 3)\n")
                f.write("~"*40 + "\n")
                f.write(record['stage3_agentspeak'])
                f.write("\n")

            elif record.get('stage3_error'):
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

    # End pipeline
    filepath = logger.end_pipeline(success=True)
    print(f"Log saved to: {filepath}")


if __name__ == "__main__":
    test_logger()
