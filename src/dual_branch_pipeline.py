"""
Dual-Branch Pipeline (DEPRECATED - Backward Compatibility Wrapper)

This file is maintained for backward compatibility only.
New code should use ltl_bdi_pipeline.py instead.

Branch B (FOND Planning) has been moved to src/legacy/fond/
The pipeline now focuses on Branch A (LLM AgentSpeak) only.
"""

import warnings
from ltl_bdi_pipeline import LTL_BDI_Pipeline


class DualBranchPipeline:
    """
    DEPRECATED: Backward compatibility wrapper for DualBranchPipeline

    This class now delegates to LTL_BDI_Pipeline and only supports
    mode="llm_agentspeak". The FOND planning mode has been moved to legacy.

    For new code, use LTL_BDI_Pipeline directly:
        from ltl_bdi_pipeline import LTL_BDI_Pipeline
        pipeline = LTL_BDI_Pipeline()
        result = pipeline.execute("Stack block A on block B")
    """

    def __init__(self):
        warnings.warn(
            "DualBranchPipeline is deprecated. Use LTL_BDI_Pipeline instead. "
            "FOND mode has been moved to src/legacy/fond/",
            DeprecationWarning,
            stacklevel=2
        )
        self._pipeline = LTL_BDI_Pipeline()
        # Expose internal attributes for compatibility
        self.config = self._pipeline.config
        self.logger = self._pipeline.logger
        self.output_dir = self._pipeline.output_dir
        self.domain_actions = self._pipeline.domain_actions
        self.domain_predicates = self._pipeline.domain_predicates

    def execute(self, nl_instruction: str, mode: str = "llm_agentspeak"):
        """
        Execute pipeline (backward compatibility method)

        Args:
            nl_instruction: Natural language instruction
            mode: Execution mode (only "llm_agentspeak" is supported)

        Returns:
            Results from execution

        Raises:
            ValueError: If mode is "fond" (no longer supported)
        """
        if mode == "fond":
            raise ValueError(
                "FOND mode is no longer supported in the main pipeline. "
                "FOND planning has been moved to src/legacy/fond/. "
                "Please use mode='llm_agentspeak' or see legacy/fond/README.md "
                "for instructions on restoring FOND functionality."
            )

        if mode != "llm_agentspeak":
            warnings.warn(
                f"Unknown mode '{mode}'. Defaulting to 'llm_agentspeak'.",
                UserWarning
            )

        # Delegate to new pipeline
        result = self._pipeline.execute(nl_instruction)

        # Update output_dir for compatibility
        self.output_dir = self._pipeline.output_dir

        # Add mode to result for compatibility
        result["mode"] = "llm_agentspeak"

        return result
