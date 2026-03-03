"""
Stage 3 PANDA planner generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stage3_code_generation.agentspeak_codegen import AgentSpeakCodeGenerator
from stage3_code_generation.htn_method_synthesis import HTNMethodSynthesizer
from stage3_code_generation.panda_planner import PANDAPlanner


class PANDAPlannerGenerator:
	"""Orchestrate Stage 3A (LLM HTN synthesis) and Stage 3B (PANDA planning)."""

	def __init__(
		self,
		domain: Any,
		grounding_map: Any,
		api_key: Optional[str] = None,
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: float = 60.0,
		workspace: Optional[str] = None,
	) -> None:
		self.domain = domain
		self.grounding_map = grounding_map
		self.synthesizer = HTNMethodSynthesizer(
			api_key=api_key,
			model=model,
			base_url=base_url,
			timeout=timeout,
		)
		self.planner = PANDAPlanner(workspace=workspace)
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
			transition_name = f"transition_{index}"
			label = literal.to_signature()
			plan = self.planner.plan(
				domain=self.domain,
				method_library=method_library,
				objects=ltl_dict.get("objects", []),
				target_literal=literal,
				transition_name=transition_name,
			)

			plan_records.append(
				{
					"transition_name": transition_name,
					"label": label,
					"target_literal": literal,
					"plan": plan,
				}
			)
			transition_artifacts.append(
				{
					"transition_name": transition_name,
					"label": label,
					"target_literal": literal.to_dict(),
					"plan": plan.to_dict(),
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
				"method": "panda",
				"backend": "pandaPI",
				"used_llm": synthesis_meta["used_llm"],
				"llm_attempted": synthesis_meta["llm_prompt"] is not None,
				"compound_tasks": len(method_library.compound_tasks),
				"primitive_tasks": len(method_library.primitive_tasks),
				"methods": len(method_library.methods),
				"transition_count": len(plan_records),
				"code_size_chars": len(agentspeak_code),
			},
			"llm": {
				"used": synthesis_meta["used_llm"],
				"model": synthesis_meta["model"],
				"prompt": synthesis_meta["llm_prompt"],
				"response": synthesis_meta["llm_response"],
			},
		}

		return agentspeak_code, artifacts
