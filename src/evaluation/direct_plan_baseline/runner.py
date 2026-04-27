from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from language_model import create_openai_compatible_client
from language_model import create_openai_compatible_json_completion
from utils.config import Config, get_config
from utils.hddl_parser import HDDLDomain, HDDLParser, HDDLProblem
from verification.official_plan_verifier import (
	IPCPlanVerifier,
	IPCPrimitivePlanVerificationResult,
)


class DirectPlanParseError(ValueError):
	"""Raised when a direct plan-generation response violates the JSON contract."""


@dataclass(frozen=True)
class DirectPlanBaselineResult:
	domain_key: str
	query_id: str
	problem_file: str
	output_dir: str
	prompt_file: str
	raw_response_file: str
	plan_file: str
	validation_file: str
	parseable: bool
	executable: bool
	goal_reached: bool
	success: bool
	diagnostics: tuple[str, ...]
	verification_skipped: bool = False
	error: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"domain_key": self.domain_key,
			"query_id": self.query_id,
			"problem_file": self.problem_file,
			"output_dir": self.output_dir,
			"prompt_file": self.prompt_file,
			"raw_response_file": self.raw_response_file,
			"plan_file": self.plan_file,
			"validation_file": self.validation_file,
			"parseable": self.parseable,
			"executable": self.executable,
			"goal_reached": self.goal_reached,
			"success": self.success,
			"diagnostics": list(self.diagnostics),
			"verification_skipped": self.verification_skipped,
			"error": self.error,
		}


def build_direct_plan_system_prompt() -> str:
	return (
		"ROLE:\n"
		"Generate a verifier-readable primitive plan artifact pi_dir for one "
		"benchmark case in the Chapter 5 direct language-model baseline.\n"
		"\n"
		"OBJECTIVE:\n"
		"Given the HDDL domain vocabulary, one bound problem instance I, and one "
		"LTLf temporal specification phi_i, produce the ordered primitive action "
		"lines that will be materialized as plan.txt and checked by the IPC/PANDA "
		"verifier for executable goal-reaching behavior.\n"
		"\n"
		"CONTRACT:\n"
		"- Use only declared primitive action names from the supplied action schemas.\n"
		"- Use only object constants from the supplied problem instance.\n"
		"- Respect action arity, argument typing, preconditions, and effects.\n"
		"- Order actions so every step is executable from the state produced by the previous steps.\n"
		"- The final state must reach the supplied problem goal and satisfy the temporal specification.\n"
		"- Generate only one case-specific final primitive plan, not HTN methods, BDI plans, schemas, proofs, markdown, or search traces.\n"
		"\n"
		"OUTPUT:\n"
		"Return exactly one minified JSON object with top-level keys plan_lines and diagnostics. "
		"plan_lines must be an array of strings. diagnostics must be an array of short strings. "
		"Do not emit markdown or commentary outside the JSON object.\n"
	)


def build_direct_plan_user_prompt(
	*,
	domain: HDDLDomain,
	problem: HDDLProblem,
	temporal_specification: Any,
	instruction: str,
) -> str:
	ltlf_formula = str(getattr(temporal_specification, "ltlf_formula", "") or "").strip()
	instruction_id = str(
		getattr(temporal_specification, "instruction_id", "") or "",
	).strip()
	referenced_events = _temporal_referenced_events(temporal_specification)
	sections = [
		_tagged_block(
			"task",
			"Generate one primitive plan for this single bound benchmark case. "
			"The evaluation wrapper will write plan_lines into plan.txt between "
			"the required '==>' header and 'root' footer.",
		),
		_tagged_block("case", f"instruction_id: {instruction_id}\ninstruction: {instruction}"),
		_tagged_block("domain_vocabulary", _render_domain_vocabulary(domain)),
		_tagged_block("problem_instance", _render_problem_instance(problem)),
		_tagged_block(
			"temporal_specification",
			"\n".join(
				[
					f"ltlf_formula: {ltlf_formula}",
					"referenced_events:",
					"\n".join(f"- {event}" for event in referenced_events) or "- none",
				],
			),
		),
		_tagged_block(
			"output_format",
			"Each plan_lines item must be one verifier primitive step without wrapper lines:\n"
			"- format: '<zero_based_index> <action_name> <arg1> <arg2> ...'\n"
			"- example: '0 pick-up b1'\n"
			"- do not include '==>' or 'root' in plan_lines\n"
			"- indexes must be contiguous from 0 in the emitted order",
		),
		_tagged_block(
			"gate_checklist",
			"Before emitting JSON, check that every action is declared, every object is "
			"declared in the problem, each action has the right arity and types, each "
			"precondition is intended to hold when the action is executed, and the final "
			"state reaches the supplied goal.",
		),
	]
	return "\n\n".join(sections)


class DirectPlanGenerator:
	"""OpenAI-compatible direct primitive-plan generator."""

	def __init__(
		self,
		*,
		config: Optional[Config] = None,
		api_key: Optional[str] = None,
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		timeout: Optional[float] = None,
		max_tokens: Optional[int] = None,
	) -> None:
		active_config = config or get_config()
		self.api_key = api_key if api_key is not None else active_config.direct_plan_generation_api_key
		self.model = model or active_config.direct_plan_generation_model
		self.base_url = (
			base_url
			if base_url is not None
			else active_config.direct_plan_generation_base_url
		)
		self.timeout = (
			float(timeout)
			if timeout is not None
			else float(active_config.direct_plan_generation_timeout)
		)
		self.max_tokens = (
			int(max_tokens)
			if max_tokens is not None
			else int(active_config.direct_plan_generation_max_tokens)
		)
		self.client = None
		if self.api_key:
			self.client = create_openai_compatible_client(
				api_key=self.api_key,
				base_url=self.base_url,
				timeout=self.timeout,
				max_retries=0,
			)

	def generate(self, *, system_prompt: str, user_prompt: str) -> tuple[str, Dict[str, Any]]:
		if self.client is None:
			raise ValueError("DIRECT_PLAN_GENERATION_API_KEY is required for API generation.")
		started_at = time.perf_counter()
		response = create_openai_compatible_json_completion(
			self.client,
			model=self.model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			max_tokens=self.max_tokens,
			timeout=self.timeout,
		)
		choice = response.choices[0]
		message = getattr(choice, "message", None)
		content = str(getattr(message, "content", "") or "")
		return content, {
			"model": self.model,
			"base_url": self.base_url,
			"timeout": self.timeout,
			"max_tokens": self.max_tokens,
			"finish_reason": getattr(choice, "finish_reason", None),
			"duration_seconds": round(time.perf_counter() - started_at, 3),
			"response_id": getattr(response, "id", None),
		}


def run_direct_plan_baseline_case(
	*,
	domain_key: str,
	query_id: str,
	domain_file: str | Path,
	problem_file: str | Path,
	instruction: str,
	temporal_specification: Any,
	output_dir: str | Path,
	response_text: Optional[str] = None,
	generator: Optional[DirectPlanGenerator] = None,
	verifier: Optional[IPCPlanVerifier] = None,
	verify: bool = True,
	system_prompt_override: Optional[str] = None,
	user_prompt_override: Optional[str] = None,
) -> DirectPlanBaselineResult:
	domain = HDDLParser.parse_domain(str(domain_file))
	problem = HDDLParser.parse_problem(str(problem_file))
	system_prompt = system_prompt_override or build_direct_plan_system_prompt()
	user_prompt = user_prompt_override or build_direct_plan_user_prompt(
		domain=domain,
		problem=problem,
		temporal_specification=temporal_specification,
		instruction=instruction,
	)

	query_output_dir = Path(output_dir).resolve()
	query_output_dir.mkdir(parents=True, exist_ok=True)
	prompt_file = query_output_dir / "prompt.json"
	raw_response_file = query_output_dir / "response.json"
	plan_file = query_output_dir / "plan.txt"
	validation_file = query_output_dir / "direct_plan_validation.json"
	prompt_payload = {
		"domain_key": domain_key,
		"query_id": query_id,
		"domain_file": str(Path(domain_file).resolve()),
		"problem_file": str(Path(problem_file).resolve()),
		"system": system_prompt,
		"user": user_prompt,
	}
	prompt_file.write_text(json.dumps(prompt_payload, indent=2))

	llm_metadata: Dict[str, Any] = {"source": "response_file" if response_text is not None else "api"}
	if response_text is None:
		active_generator = generator or DirectPlanGenerator()
		response_text, llm_metadata = active_generator.generate(
			system_prompt=system_prompt,
			user_prompt=user_prompt,
		)
	raw_response_file.write_text(
		json.dumps(
			{
				"response_text": response_text,
				"llm": llm_metadata,
			},
			indent=2,
		),
	)

	try:
		parsed = parse_direct_plan_response(str(response_text or ""))
		plan_text = materialize_plan_text(parsed["plan_lines"])
		plan_file.write_text(plan_text)
		if not verify:
			result = DirectPlanBaselineResult(
				domain_key=domain_key,
				query_id=query_id,
				problem_file=str(Path(problem_file).resolve()),
				output_dir=str(query_output_dir),
				prompt_file=str(prompt_file),
				raw_response_file=str(raw_response_file),
				plan_file=str(plan_file),
				validation_file=str(validation_file),
				parseable=True,
				executable=False,
				goal_reached=False,
				success=False,
				diagnostics=tuple(parsed["diagnostics"]),
				verification_skipped=True,
				error=None,
			)
			validation_file.write_text(json.dumps(result.to_dict(), indent=2))
			return result
		verification_result = (verifier or IPCPlanVerifier()).verify_plan_text(
			domain_file=domain_file,
			problem_file=problem_file,
			plan_text=plan_text,
			output_dir=query_output_dir,
			plan_kind="primitive_only",
			plan_filename="plan.txt",
			output_filename="verifier.txt",
			json_filename="verifier.json",
		)
		result = _build_result(
			domain_key=domain_key,
			query_id=query_id,
			problem_file=problem_file,
			output_dir=query_output_dir,
			prompt_file=prompt_file,
			raw_response_file=raw_response_file,
			plan_file=plan_file,
			validation_file=validation_file,
			parseable=True,
			diagnostics=tuple(parsed["diagnostics"]),
			verification_result=verification_result,
		)
	except Exception as exc:
		plan_file.write_text("")
		result = DirectPlanBaselineResult(
			domain_key=domain_key,
			query_id=query_id,
			problem_file=str(Path(problem_file).resolve()),
			output_dir=str(query_output_dir),
			prompt_file=str(prompt_file),
			raw_response_file=str(raw_response_file),
			plan_file=str(plan_file),
			validation_file=str(validation_file),
			parseable=False,
			executable=False,
			goal_reached=False,
			success=False,
			diagnostics=(),
			verification_skipped=not verify,
			error=str(exc),
		)
	validation_file.write_text(json.dumps(result.to_dict(), indent=2))
	return result


def parse_direct_plan_response(response_text: str) -> Dict[str, Any]:
	payload_text = _extract_json_object(response_text)
	try:
		payload = json.loads(payload_text)
	except json.JSONDecodeError as exc:
		raise DirectPlanParseError(f"Direct plan response is not valid JSON: {exc}") from exc
	if not isinstance(payload, dict):
		raise DirectPlanParseError("Direct plan response must be a JSON object.")
	unknown_keys = set(payload) - {"plan_lines", "diagnostics"}
	if unknown_keys:
		raise DirectPlanParseError(
			"Direct plan response may contain only plan_lines and diagnostics; "
			f"unexpected keys: {sorted(unknown_keys)}",
		)
	plan_lines = payload.get("plan_lines")
	diagnostics = payload.get("diagnostics", [])
	if not isinstance(plan_lines, list) or not all(
		isinstance(line, str) for line in plan_lines
	):
		raise DirectPlanParseError("plan_lines must be an array of strings.")
	if not isinstance(diagnostics, list) or not all(
		isinstance(item, str) for item in diagnostics
	):
		raise DirectPlanParseError("diagnostics must be an array of strings.")
	return {
		"plan_lines": [line.strip() for line in plan_lines if line.strip()],
		"diagnostics": [item.strip() for item in diagnostics if item.strip()],
	}


def materialize_plan_text(plan_lines: Sequence[str]) -> str:
	body_lines = []
	for raw_line in plan_lines:
		line = str(raw_line or "").strip()
		if not line or line in {"==>", "root"}:
			continue
		body_lines.append(line)
	return "\n".join(["==>", *body_lines, "root"]) + "\n"


def _build_result(
	*,
	domain_key: str,
	query_id: str,
	problem_file: str | Path,
	output_dir: Path,
	prompt_file: Path,
	raw_response_file: Path,
	plan_file: Path,
	validation_file: Path,
	parseable: bool,
	diagnostics: Sequence[str],
	verification_result: IPCPrimitivePlanVerificationResult,
) -> DirectPlanBaselineResult:
	executable = verification_result.primitive_plan_executable is True
	goal_reached = verification_result.reached_goal_state is True
	success = parseable and executable and goal_reached
	return DirectPlanBaselineResult(
		domain_key=domain_key,
		query_id=query_id,
		problem_file=str(Path(problem_file).resolve()),
		output_dir=str(output_dir),
		prompt_file=str(prompt_file),
		raw_response_file=str(raw_response_file),
		plan_file=str(plan_file),
		validation_file=str(validation_file),
		parseable=parseable,
		executable=executable,
		goal_reached=goal_reached,
		success=success,
		diagnostics=tuple(diagnostics),
		error=None if success else verification_result.error,
	)


def _extract_json_object(response_text: str) -> str:
	text = str(response_text or "").strip()
	if text.startswith("{") and text.endswith("}"):
		return text
	start = text.find("{")
	end = text.rfind("}")
	if start < 0 or end <= start:
		raise DirectPlanParseError("Direct plan response did not contain a JSON object.")
	return text[start : end + 1]


def _tagged_block(tag: str, body: str) -> str:
	return f"<{tag}>\n{body.strip()}\n</{tag}>"


def _render_domain_vocabulary(domain: HDDLDomain) -> str:
	lines = [
		f"domain: {domain.name}",
		"types:",
		*(f"- {item}" for item in domain.types),
		"predicates:",
		*(f"- {predicate.to_signature()}" for predicate in domain.predicates),
		"primitive_action_schemas:",
	]
	for action in domain.actions:
		lines.extend(
			[
				f"- action: {action.name}",
				f"  parameters: {', '.join(action.parameters) or 'none'}",
				f"  precondition: {action.preconditions or '()'}",
				f"  effect: {action.effects or '()'}",
			],
		)
	return "\n".join(lines)


def _render_problem_instance(problem: HDDLProblem) -> str:
	object_lines = [
		f"- {object_name}: {problem.object_types.get(object_name, 'object')}"
		for object_name in problem.objects
	]
	init_lines = [f"- {fact.to_signature()}" for fact in problem.init_facts]
	goal_lines = [f"- {fact.to_signature()}" for fact in problem.goal_facts]
	root_task_lines = [f"- {task.to_signature()}" for task in problem.htn_tasks]
	return "\n".join(
		[
			f"problem: {problem.name}",
			f"domain: {problem.domain_name}",
			"objects:",
			"\n".join(object_lines) or "- none",
			"initial_state:",
			"\n".join(init_lines) or "- none",
			"goal_condition:",
			"\n".join(goal_lines) or "- none",
			"root_htn_tasks_for_temporal_context:",
			"\n".join(root_task_lines) or "- none",
		],
	)


def _temporal_referenced_events(temporal_specification: Any) -> tuple[str, ...]:
	events = []
	for event in getattr(temporal_specification, "referenced_events", ()) or ():
		if isinstance(event, dict):
			event_text = str(event.get("event") or "").strip()
		else:
			event_text = str(getattr(event, "event", "") or "").strip()
		if event_text:
			events.append(event_text)
	if events:
		return tuple(events)
	ltlf_formula = str(getattr(temporal_specification, "ltlf_formula", "") or "")
	return tuple(
		match.group(0)
		for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_-]*\([^()]*\)", ltlf_formula)
	)
