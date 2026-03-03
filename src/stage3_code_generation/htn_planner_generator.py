"""
Stage 3 HTN planner generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stage3_code_generation.agentspeak_codegen import AgentSpeakCodeGenerator
from stage3_code_generation.htn_method_synthesis import HTNMethodSynthesizer
from stage3_code_generation.htn_planner import HTNPlanner
from stage3_code_generation.htn_schema import HTNLiteral
from stage3_code_generation.htn_specialiser import HTNSpecialiser


class HTNPlannerGenerator:
    """Orchestrate Stage 3A-3D for the HTN refactor."""

    def __init__(
        self,
        domain: Any,
        grounding_map: Any,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.domain = domain
        self.grounding_map = grounding_map
        self.synthesizer = HTNMethodSynthesizer(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            client=client,
        )
        self.planner = HTNPlanner()
        self.specialiser = HTNSpecialiser()
        self.codegen = AgentSpeakCodeGenerator()

    def generate(self, ltl_dict: Dict[str, Any], dfa_result: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        grounding_map = ltl_dict.get("grounding_map", self.grounding_map)
        method_library, synthesis_meta = self.synthesizer.synthesize(
            domain=self.domain,
            grounding_map=grounding_map,
            dfa_result=dfa_result,
        )

        plan_records: List[Dict[str, Any]] = []
        transition_artifacts: List[Dict[str, Any]] = []

        for index, literal in enumerate(method_library.target_literals, start=1):
            task_name = self._task_name_for_literal(literal)
            trace = self.planner.build_trace(
                library=method_library,
                task_name=task_name,
                args=literal.args,
                literal=literal,
            )
            specialisation = self.specialiser.specialise(trace, literal)
            transition_name = f"transition_{index}"
            label = literal.to_signature()

            plan_records.append(
                {
                    "transition_name": transition_name,
                    "label": label,
                    "target_literal": literal,
                    "trace": trace,
                    "specialisation": specialisation,
                }
            )
            transition_artifacts.append(
                {
                    "transition_name": transition_name,
                    "label": label,
                    "target_literal": literal.to_dict(),
                    "trace": trace.to_dict(),
                    "specialisation": specialisation.to_dict(),
                }
            )

        agentspeak_code = self.codegen.generate(
            domain=self.domain,
            objects=ltl_dict.get("objects", []),
            plan_records=plan_records,
        )

        artifacts = {
            "method_library": method_library.to_dict(),
            "transitions": transition_artifacts,
            "summary": {
                "method": "htn",
                "used_llm": synthesis_meta["used_llm"],
                "llm_attempted": synthesis_meta["llm_prompt"] is not None,
                "compound_tasks": len(method_library.compound_tasks),
                "primitive_tasks": len(method_library.primitive_tasks),
                "methods": len(method_library.methods),
                "transition_count": len(plan_records),
                "code_size_chars": len(agentspeak_code),
                "fallback_reason": synthesis_meta["fallback_reason"],
            },
            "llm": {
                "used": synthesis_meta["used_llm"],
                "model": synthesis_meta["model"],
                "prompt": synthesis_meta["llm_prompt"],
                "response": synthesis_meta["llm_response"],
            },
        }

        return agentspeak_code, artifacts

    @staticmethod
    def _task_name_for_literal(literal: HTNLiteral) -> str:
        predicate = literal.predicate.replace("-", "_")
        if literal.is_positive:
            return f"achieve_{predicate}"
        return f"maintain_not_{predicate}"
