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
    stage2_num_states: int = 0  # Simplified DFA states
    stage2_num_transitions: int = 0  # Simplified DFA transitions
    stage2_original_num_states: int = 0  # Original DFA states (before simplification)
    stage2_original_num_transitions: int = 0  # Original DFA transitions (before simplification)
    stage2_error: Optional[str] = None

    # Stage 3: DFAs -> AgentSpeak Code Generation
    stage3_status: str = "pending"
    stage3_agentspeak: Optional[str] = None
    stage3_error: Optional[str] = None
    stage3_used_llm: bool = False
    stage3_model: Optional[str] = None
    stage3_llm_prompt: Optional[Dict[str, str]] = None
    stage3_llm_response: Optional[str] = None

    # Stage 3: Backward Planning Statistics (non-LLM method)
    stage3_method: str = "llm"  # "llm" or "backward_planning"
    stage3_states_explored: int = 0
    stage3_transitions_generated: int = 0
    stage3_goal_plans_count: int = 0
    stage3_action_plans_count: int = 0
    stage3_cache_hits: int = 0
    stage3_cache_misses: int = 0
    stage3_ground_actions_cached: int = 0
    stage3_code_size_chars: int = 0
    stage3_redundancy_eliminated_pct: float = 0.0

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
            # NEW: Log original DFA statistics
            self.current_record.stage2_original_num_states = dfa_result.get('original_num_states', 0)
            self.current_record.stage2_original_num_transitions = dfa_result.get('original_num_transitions', 0)

            # Save both original and simplified DFA DOT files
            if self.current_log_dir:
                # Save original DFA (before simplification)
                if 'original_dfa_dot' in dfa_result:
                    original_dfa_path = self.current_log_dir / "dfa_original.dot"
                    with open(original_dfa_path, 'w') as f:
                        f.write(dfa_result['original_dfa_dot'])

                # Save simplified DFA (after simplification)
                if 'dfa_dot' in dfa_result:
                    simplified_dfa_path = self.current_log_dir / "dfa_simplified.dot"
                    with open(simplified_dfa_path, 'w') as f:
                        f.write(dfa_result['dfa_dot'])
        elif error:
            self.current_record.stage2_status = "failed"
            self.current_record.stage2_error = str(error)

        # IMMEDIATELY save current state to files
        self._save_current_state()

    # NEW: Stage 3 - AgentSpeak Generation logging
    def log_stage3(self, ltl_spec: Any, dfa_result: Any, agentspeak_code: str, status: str,
                   error: str = None, model: str = None, llm_prompt: Dict[str, str] = None,
                   llm_response: str = None,
                   # Backward planning statistics
                   method: str = "llm", states_explored: int = 0, transitions_generated: int = 0,
                   goal_plans_count: int = 0, action_plans_count: int = 0,
                   cache_hits: int = 0, cache_misses: int = 0, ground_actions_cached: int = 0,
                   redundancy_eliminated_pct: float = 0.0):
        """
        Log Stage 3: DFA -> AgentSpeak Code Generation

        Args:
            ltl_spec: LTL specification
            dfa_result: DFA result dictionary
            agentspeak_code: Generated AgentSpeak code (MUST be complete)
            status: "Success" or "Failed"
            error: Error message if failed
            model: LLM model name (if using LLM method)
            llm_prompt: LLM prompt dict (if using LLM method)
            llm_response: LLM response text (if using LLM method)
            method: "llm" or "backward_planning"
            states_explored: Number of states explored (backward planning)
            transitions_generated: Number of transitions generated
            goal_plans_count: Number of goal achievement plans generated
            action_plans_count: Number of action plans generated
            cache_hits: Number of goal exploration cache hits
            cache_misses: Number of goal exploration cache misses
            ground_actions_cached: Number of ground actions cached
            redundancy_eliminated_pct: Percentage of redundancy eliminated
        """
        if not self.current_record:
            return

        # Set generation method
        self.current_record.stage3_method = method

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
            self.current_record.stage3_code_size_chars = len(agentspeak_code)

            # Save AgentSpeak code to .asl file
            if self.current_log_dir:
                asl_filepath = self.current_log_dir / "generated_code.asl"
                with open(asl_filepath, 'w') as f:
                    f.write(agentspeak_code)

        elif error:
            self.current_record.stage3_status = "failed"
            self.current_record.stage3_error = str(error)

        # Log backward planning statistics
        if method == "backward_planning":
            self.current_record.stage3_states_explored = states_explored
            self.current_record.stage3_transitions_generated = transitions_generated
            self.current_record.stage3_goal_plans_count = goal_plans_count
            self.current_record.stage3_action_plans_count = action_plans_count
            self.current_record.stage3_cache_hits = cache_hits
            self.current_record.stage3_cache_misses = cache_misses
            self.current_record.stage3_ground_actions_cached = ground_actions_cached
            self.current_record.stage3_redundancy_eliminated_pct = redundancy_eliminated_pct

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

                f.write(f"Formula: {record.get('stage2_formula', 'N/A')}\n\n")

                # Show both original and simplified DFA statistics
                f.write("Original DFA (before simplification):\n")
                f.write(f"  States: {record.get('stage2_original_num_states', 0)}\n")
                f.write(f"  Transitions: {record.get('stage2_original_num_transitions', 0)}\n")
                f.write(f"  File: dfa_original.dot\n\n")

                f.write("Simplified DFA (after simplification):\n")
                f.write(f"  States: {record.get('stage2_num_states', 0)}\n")
                f.write(f"  Transitions: {record.get('stage2_num_transitions', 0)}\n")
                f.write(f"  File: dfa_simplified.dot\n\n")

                # Show simplification statistics if available
                dfa_result = record['stage2_dfa_result']
                if 'simplification_stats' in dfa_result:
                    stats = dfa_result['simplification_stats']
                    f.write("Simplification Statistics:\n")
                    f.write(f"  Method: {stats.get('method', 'N/A')}\n")
                    f.write(f"  Predicates: {stats.get('num_predicates', 0)}\n")
                    if 'num_original_states' in stats:
                        f.write(f"  Original States → New States: {stats['num_original_states']} → {stats['num_new_states']}\n")
                        f.write(f"  Original Transitions → New Transitions: {stats['num_original_transitions']} → {stats['num_new_transitions']}\n")
                    f.write("\n")

                # Optionally show a snippet of the simplified DFA DOT format
                dfa_dot = dfa_result.get('dfa_dot', '')
                if dfa_dot:
                    f.write(f"Simplified DFA DOT Format (first 300 chars):\n")
                    f.write(dfa_dot[:300] + "...\n")

                f.write("\n")

            elif record.get('stage2_error'):
                f.write(f"\nError: {record['stage2_error']}\n")

            f.write("\n")

            # Stage 3: AgentSpeak Generation
            f.write("-"*80 + "\n")
            f.write("STAGE 3: DFA → AgentSpeak Code Generation\n")
            f.write("-"*80 + "\n")
            f.write(f"Status: {record.get('stage3_status', 'pending').upper()}\n")
            f.write(f"Method: {record.get('stage3_method', 'N/A').upper()}\n")

            if record.get('stage3_used_llm'):
                f.write(f"Generator: LLM ({record.get('stage3_model', 'N/A')})\n")
            elif record.get('stage3_method') == 'backward_planning':
                f.write("Generator: Backward Planning (non-LLM)\n")

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

            # Backward Planning Statistics
            if record.get('stage3_method') == 'backward_planning' and record.get('stage3_status') == 'success':
                f.write("\n" + "~"*40 + "\n")
                f.write("BACKWARD PLANNING STATISTICS\n")
                f.write("~"*40 + "\n")
                f.write(f"States Explored: {record.get('stage3_states_explored', 0):,}\n")
                f.write(f"Transitions Generated: {record.get('stage3_transitions_generated', 0):,}\n")
                f.write(f"Goal Plans: {record.get('stage3_goal_plans_count', 0)}\n")
                f.write(f"Action Plans: {record.get('stage3_action_plans_count', 0)}\n")
                f.write(f"\nOptimization Metrics:\n")
                f.write(f"  Ground Actions Cached: {record.get('stage3_ground_actions_cached', 0)}\n")
                f.write(f"  Goal Cache Hits: {record.get('stage3_cache_hits', 0)}\n")
                f.write(f"  Goal Cache Misses: {record.get('stage3_cache_misses', 0)}\n")
                total_goals = record.get('stage3_cache_hits', 0) + record.get('stage3_cache_misses', 0)
                if total_goals > 0:
                    hit_rate = record.get('stage3_cache_hits', 0) / total_goals * 100
                    f.write(f"  Cache Hit Rate: {hit_rate:.1f}%\n")
                f.write(f"  Code Redundancy Eliminated: {record.get('stage3_redundancy_eliminated_pct', 0):.1f}%\n")
                f.write(f"\nCode Size: {record.get('stage3_code_size_chars', 0):,} characters\n")
                f.write("\n")

            if record.get('stage3_status') == 'success' and record.get('stage3_agentspeak'):
                f.write("\n" + "~"*40 + "\n")
                f.write("GENERATED AGENTSPEAK CODE (Stage 3)\n")
                f.write("~"*40 + "\n")
                f.write("NOTE: Complete AgentSpeak code saved to generated_code.asl\n")
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
