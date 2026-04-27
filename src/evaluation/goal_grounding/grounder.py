"""
LLM-driven temporal goal grounding for the Jason evaluation runtime.
"""

from __future__ import annotations

import json
import re
import signal
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

from method_library.synthesis.schema import HTNMethodLibrary
from evaluation.artifacts import GroundedSubgoal, TemporalGroundingResult
from evaluation.goal_grounding.canonical_ordered_formula import (
	CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE,
	apply_task_event_occurrence_suffixes,
	build_ordered_benchmark_formula,
	build_unordered_eventuality_formula,
	ordered_formula_style_prompt_guidance,
)
from language_model import (
	OPENAI_COMPATIBLE_JSON_PROFILE_NAME,
	create_openai_compatible_client,
	create_openai_compatible_json_completion,
)
from utils.config import DEFAULT_LTLF_GENERATION_MODEL
from utils.config import DEFAULT_LTLF_GENERATION_SESSION_ID
from utils.config import DEFAULT_LTLF_GENERATION_TIMEOUT_SECONDS
from utils.symbol_normalizer import SymbolNormalizer

GOAL_GROUNDING_MAX_TRANSPORT_RETRIES = 3
GOAL_GROUNDING_RETRYABLE_ERROR_FRAGMENTS = (
	"exceeded the configured wall-clock timeout before a response chunk was created",
	"exceeded the configured first-chunk deadline before any streaming content arrived",
	"exceeded the configured first-chunk deadline before any streaming chunk arrived",
	"llm response did not include any choices",
	"llm response choice did not include a message payload",
	"llm response did not contain any textual completion content",
	"llm response returned empty textual completion content",
)
GOAL_GROUNDING_RETRYABLE_ERROR_CLASS_NAMES = {
	"timeouterror",
	"apitimeouterror",
	"apiconnectionerror",
	"readtimeout",
	"connecttimeout",
	"pooltimeout",
	"readerror",
	"connecterror",
	"remoteprotocolerror",
	"networkerror",
}


class GoalGroundingEmptyResponseError(RuntimeError):
	"""The provider returned no usable completion text for a grounding request."""


class GoalGroundingMalformedResponseError(ValueError):
	"""The model returned text, but it was not valid goal-grounding JSON."""


class GoalGroundingProviderUnavailable(RuntimeError):
	"""All retry attempts were exhausted before the provider returned usable text."""

	def __init__(
		self,
		message: str,
		*,
		attempt_count: int,
		attempt_errors: Sequence[Dict[str, str]],
		last_error: Exception,
	) -> None:
		super().__init__(message)
		self.attempt_count = attempt_count
		self.attempt_errors = tuple(dict(error) for error in attempt_errors)
		self.last_error = last_error


class NLToLTLfGenerator:
	"""
	Convert a natural-language query into one grounded LTLf formula.
	"""

	_FORMULA_OPERATOR_SEQUENCE_PATTERN = re.compile(r"^(?:WX|[FGXURW])+$")
	_TASK_EVENT_SUFFIX_PATTERN = re.compile(
		r"^(?P<base>[a-z_][a-z0-9_]*?)__(?:e|event)(?P<index>[1-9][0-9]*)$",
	)
	_RESERVED_FORMULA_TOKENS = {
		"true",
		"false",
		"F",
		"G",
		"X",
		"WX",
		"U",
		"R",
	}

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: Optional[str] = None,
		base_url: Optional[str] = None,
		domain_file: Optional[str] = None,
		request_timeout: Optional[float] = None,
		response_max_tokens: Optional[int] = None,
		session_id: Optional[str] = None,
	) -> None:
		self.api_key = api_key
		self.model = model or DEFAULT_LTLF_GENERATION_MODEL
		self.base_url = base_url
		self.domain_file = domain_file
		self.request_timeout = float(request_timeout or DEFAULT_LTLF_GENERATION_TIMEOUT_SECONDS)
		self.response_max_tokens = int(response_max_tokens or 12000)
		self.session_id = str(session_id or DEFAULT_LTLF_GENERATION_SESSION_ID).strip()
		self.client = None
		self.last_generation_metadata: Dict[str, Any] = {}
		self.symbol_normalizer = SymbolNormalizer()

		self.domain = None
		if domain_file:
			from utils.hddl_parser import HDDLParser

			self.domain = HDDLParser.parse_domain(domain_file)

		if api_key:
			self.client = create_openai_compatible_client(
				api_key=api_key,
				base_url=base_url,
				timeout=self.request_timeout,
				max_retries=0,
			)

	def generate(
		self,
		nl_instruction: str,
		*,
		method_library: Optional[HTNMethodLibrary] = None,
		typed_objects: Optional[Dict[str, str]] = None,
		task_type_map: Optional[Dict[str, Tuple[str, ...]]] = None,
		type_parent_map: Optional[Dict[str, Optional[str]]] = None,
	) -> Tuple[TemporalGroundingResult, Dict[str, str], str]:
		query_text = str(nl_instruction or "").strip()
		if not query_text:
			raise ValueError("Natural-language query is empty.")
		if self.domain is None:
			raise RuntimeError("NLToLTLfGenerator requires parsed domain context.")

		if not self.client:
			raise RuntimeError(
				"No API key configured. Please set LTLF_GENERATION_API_KEY in .env file.",
			)

		prompt_attempts = self._build_prompt_attempts(
			query_text=query_text,
			method_library=method_library,
			typed_objects=typed_objects or {},
			task_type_map=task_type_map or {},
		)
		current_attempt = dict(prompt_attempts[0])
		last_error: Exception | None = None
		last_response_text = ""
		last_finish_reason = ""
		last_prompt: Dict[str, str] = {
			"system": str(current_attempt["system"]),
			"user": str(current_attempt["user"]),
		}
		attempt_errors: list[Dict[str, str]] = []
		response_transport_metadata: Dict[str, Any] = {}
		result = None
		attempt_count = 0
		while True:
			attempt_count += 1
			last_response_text = ""
			last_finish_reason = ""
			response_transport_metadata = {}
			try:
				messages = [
					{"role": "system", "content": last_prompt["system"]},
					{"role": "user", "content": last_prompt["user"]},
				]
				request_profile = self._goal_grounding_request_profile(messages=messages)
				response = self._create_chat_completion(
					messages,
					response_max_tokens=int(current_attempt["response_max_tokens"]),
					request_timeout=float(current_attempt["request_timeout"]),
					request_profile=request_profile,
				)
				response_text, finish_reason, response_transport_metadata = (
					self._read_response_payload(
						response,
						request_timeout=float(current_attempt["request_timeout"]),
						transport_metadata=self._goal_grounding_transport_metadata(request_profile),
					)
				)
				last_finish_reason = finish_reason
				last_response_text = response_text
				payload = self._parse_json_blob(response_text)
				result = self._validate_payload(
					query_text=query_text,
					payload=payload,
					method_library=method_library,
					typed_objects=typed_objects or {},
					task_type_map=task_type_map or {},
					type_parent_map=type_parent_map or {},
				)
				self.last_generation_metadata = {
					"attempt_mode": str(current_attempt["mode"]),
					"attempt_count": attempt_count,
					"max_transport_retries": GOAL_GROUNDING_MAX_TRANSPORT_RETRIES,
					"attempt_errors": attempt_errors,
					"task_event_count": len(result.subgoals),
					"formula_atom_count": len(self._extract_formula_atoms(result.ltlf_formula)),
					"model": self.model,
					"last_prompt": dict(last_prompt),
					"last_response": response_text,
					"last_finish_reason": finish_reason,
					**response_transport_metadata,
				}
				return result, last_prompt, response_text
			except Exception as exc:
				response_transport_metadata = dict(
					getattr(exc, "transport_metadata", None) or response_transport_metadata or {},
				)
				transport_finish_reason = str(
					response_transport_metadata.get("llm_finish_reason") or "",
				).strip()
				if transport_finish_reason and not last_finish_reason:
					last_finish_reason = transport_finish_reason
				last_error = exc
				retryable_error = self._is_retryable_goal_grounding_error(exc)
				attempt_errors.append(
					{
						"mode": str(current_attempt["mode"]),
						"error": str(exc),
						"finish_reason": last_finish_reason,
						"retryable": "true" if retryable_error else "false",
					},
				)
				self.last_generation_metadata = {
					"attempt_mode": str(current_attempt["mode"]),
					"attempt_count": attempt_count,
					"max_transport_retries": GOAL_GROUNDING_MAX_TRANSPORT_RETRIES,
					"attempt_errors": list(attempt_errors),
					"model": self.model,
					"last_prompt": dict(last_prompt),
					"last_response": last_response_text,
					"last_finish_reason": last_finish_reason,
					**response_transport_metadata,
				}
				if retryable_error and attempt_count <= GOAL_GROUNDING_MAX_TRANSPORT_RETRIES:
					continue
				break
		if last_error is None:
			raise RuntimeError("Goal grounding failed without a recorded exception.")
		self.last_generation_metadata = {
			"attempt_mode": str(current_attempt["mode"]),
			"attempt_count": attempt_count,
			"max_transport_retries": GOAL_GROUNDING_MAX_TRANSPORT_RETRIES,
			"attempt_errors": attempt_errors,
			"model": self.model,
			"last_prompt": dict(last_prompt),
			"last_response": last_response_text,
			"last_response_preview": last_response_text[:500],
			"last_finish_reason": last_finish_reason,
			**response_transport_metadata,
		}
		if self._is_retryable_goal_grounding_error(last_error):
			provider_error = GoalGroundingProviderUnavailable(
				"Goal-grounding provider did not return usable completion text after "
				f"{attempt_count} attempts: {last_error}",
				attempt_count=attempt_count,
				attempt_errors=attempt_errors,
				last_error=last_error,
			)
			try:
				setattr(
					provider_error,
					"transport_metadata",
					dict(getattr(last_error, "transport_metadata", {}) or {}),
				)
			except Exception:
				pass
			raise provider_error from last_error
		raise last_error

	@staticmethod
	def _is_retryable_goal_grounding_error(exc: Exception) -> bool:
		error_class_name = exc.__class__.__name__.strip().lower()
		if error_class_name in GOAL_GROUNDING_RETRYABLE_ERROR_CLASS_NAMES:
			return True
		error_message = str(exc or "").strip().lower()
		return any(
			fragment in error_message
			for fragment in GOAL_GROUNDING_RETRYABLE_ERROR_FRAGMENTS
		)

	def _build_prompt_attempts(
		self,
		*,
		query_text: str,
		method_library: Optional[HTNMethodLibrary],
		typed_objects: Dict[str, str],
		task_type_map: Dict[str, Tuple[str, ...]],
	) -> Tuple[Dict[str, Any], ...]:
		standard_system, standard_user = self._build_prompts(
			query_text=query_text,
			method_library=method_library,
			typed_objects=typed_objects,
			task_type_map=task_type_map,
		)
		return tuple(
			{
				"index": index,
				"mode": mode,
				"system": system_prompt,
				"user": user_prompt,
				"response_max_tokens": response_max_tokens,
				"request_timeout": current_request_timeout,
			}
			for index, (mode, system_prompt, user_prompt, response_max_tokens, current_request_timeout) in enumerate(
				(
					(
						"few_shot_strict",
						standard_system,
						standard_user,
						max(int(self.response_max_tokens), self._suggest_response_max_tokens(query_text)),
						max(float(self.request_timeout), self._suggest_request_timeout(query_text)),
					),
				),
				start=1,
			)
		)

	def _build_prompts(
		self,
		*,
		query_text: str,
		method_library: Optional[HTNMethodLibrary],
		typed_objects: Dict[str, str],
		task_type_map: Dict[str, Tuple[str, ...]],
	) -> Tuple[str, str]:
		task_lines = self._task_inventory_lines(method_library, task_type_map)
		object_lines = self._prompt_object_inventory_lines(query_text, typed_objects)
		few_shot_lines = self._few_shot_prompt_lines(
			typed_objects=typed_objects,
			task_type_map=task_type_map,
		)
		domain_task_signatures = "\n".join(
			f"- {task.to_signature()}"
			for task in getattr(self.domain, "tasks", ())
		)
		system_prompt = "\n".join(
			[
				"You convert natural-language planning queries into one grounded LTLf formula.",
				"You must not explain your reasoning. Return minified JSON only.",
				"Preserve temporal meaning exactly.",
				"If the query states an explicit order such as first, then, before, after, next, or finally, the LTLf formula must encode that order explicitly.",
				"Do not collapse an explicitly ordered task list into an unordered conjunction of independent eventualities.",
				f"Domain: {self.domain.name}",
				"",
				"Declared HDDL task signatures:",
				domain_task_signatures or "- none",
				"",
				"Callable grounded evaluation tasks:",
				task_lines or "- none",
				"",
				"Grounded problem objects relevant to this query:",
				object_lines or "- none",
				"",
				"Output contract:",
				'- Return exactly one JSON object with the single key "ltlf_formula".',
				'- "ltlf_formula" must be a single LTLf formula string.',
				'- Output must be minified JSON with no markdown fences, no prose, and no extra keys.',
				"- MUST: the final JSON answer must appear in the completion response content itself.",
				"- MUST NOT leave the final answer only in hidden reasoning content.",
				"- MUST NOT return an empty completion response.",
				"- Every atomic proposition in ltlf_formula must itself be one grounded task-event call.",
				"- Every atomic proposition must reuse an exact callable task name from the provided inventory.",
				"- Never add or remove a task-name prefix such as do_.",
				"- If the same grounded task call must occur multiple times, each repeated occurrence must use a distinct grounded task-event identity by appending an occurrence suffix to the task name, such as task__e1(arg1, arg2) and task__e2(arg1, arg2).",
				"- Never reuse the exact same task-event atom twice in one formula.",
				"- Do not use placeholder atoms such as subgoal_1, query_step_1, goal_1, or event_1.",
				'- Do not output lifted variables such as ?x. All args must be grounded problem objects from the current problem.',
				"- For large problems, the prompt may list only objects explicitly named in the query; validation still checks every argument against the full problem object set.",
				'- Do not invent tasks or objects.',
				"- Do not derive new task names from object inventories, setup inventories, or domain descriptions.",
				"- If the query begins with an object inventory such as 'Using ...' or any other setup inventory, treat that inventory as context only, not as task-event atoms.",
				"- Create task-event atoms only from explicitly requested grounded task invocations in the query.",
				"- Do not create task-event atoms from bare object names, setup inventories, or explanatory text.",
				"- If the query literally lists grounded task calls after a phrase like 'complete the tasks', copy exactly those task calls into the formula and keep their order.",
				"- If that listed task sequence repeats the same grounded task call multiple times, preserve the repeated order and count, but distinguish each repeated occurrence with a unique __eN suffix on the task name.",
				"- Treat a comma-separated task list after 'complete the tasks' as ordered by default. Preserve the listed order in ltlf_formula unless the query explicitly states that order does not matter or may be arbitrary.",
				"- If the query is ambiguous, still return your best grounded interpretation as one formula. Do not add extra fields.",
				'- The formula string must not contain JSON braces "{" or "}".',
				"",
				"Formula syntax reminders:",
				"- Atoms: grounded task-event calls using exact callable inventory names.",
				"- Repeated grounded task calls must use occurrence-tagged atoms such as task__e1(...) and task__e2(...).",
				"- Supported unary operators: F, G, X, WX",
				"- Supported binary operators: U, R",
				"- Supported Boolean operators: !, &, |, ->, <->",
				"- Supported constants and position predicates: true, false, last",
				"- Do not use unsupported past-time operators such as Y, WY, O, S, P, or H.",
				*ordered_formula_style_prompt_guidance(
					style=CANONICAL_BENCHMARK_ORDERED_FORMULA_STYLE,
				),
				"- Use a plain conjunction of eventualities only when the query does not impose an order.",
				"",
				"Few-shot examples:",
				*few_shot_lines,
			],
		)
		user_prompt = "\n".join(
			[
				"Ground this query into one LTLf formula only.",
				f'Query: "{query_text}"',
			],
		)
		return system_prompt, user_prompt

	def _task_inventory_lines(
		self,
		method_library: Optional[HTNMethodLibrary],
		task_type_map: Dict[str, Tuple[str, ...]],
	) -> str:
		lines = []
		if method_library is None:
			for task_name, arg_types in sorted(task_type_map.items()):
				task_name_text = str(task_name or "").strip()
				if not task_name_text:
					continue
				typed_signature = ", ".join(arg_types) if arg_types else "untyped"
				lines.append(f"- {task_name_text}({typed_signature})")
			return "\n".join(dict.fromkeys(lines))

		for task in tuple(method_library.compound_tasks or ()) + tuple(method_library.primitive_tasks or ()):
			task_name = str(getattr(task, "name", "") or "").strip()
			if not task_name:
				continue
			source_name = str(getattr(task, "source_name", "") or "").strip()
			arg_types = task_type_map.get(task_name, ())
			typed_signature = ", ".join(arg_types) if arg_types else "untyped"
			source_suffix = f" | source_name={source_name}" if source_name and source_name != task_name else ""
			lines.append(f"- {task_name}({typed_signature}){source_suffix}")
		return "\n".join(lines)

	def _few_shot_prompt_lines(
		self,
		*,
		typed_objects: Dict[str, str],
		task_type_map: Dict[str, Tuple[str, ...]],
	) -> Tuple[str, ...]:
		task_names = tuple(
			task_name
			for task_name in task_type_map.keys()
			if str(task_name).strip()
		)
		if not task_names:
			return ()
		task_name_set = {str(task_name).strip() for task_name in task_names}
		selected_task_names = self._few_shot_task_family(task_name_set) or task_names[:3]
		if len(selected_task_names) == 1:
			selected_task_names = (*selected_task_names, selected_task_names[0])

		unordered_atoms = tuple(
			self._sample_grounded_task_call(
				task_name=task_name,
				task_type_map=task_type_map,
				typed_objects=typed_objects,
			)
			for task_name in selected_task_names[:2]
		)
		ordered_atoms = tuple(
			self._sample_grounded_task_call(
				task_name=task_name,
				task_type_map=task_type_map,
				typed_objects=typed_objects,
			)
			for task_name in selected_task_names[: min(3, len(selected_task_names))]
		)
		repeated_atom = self._sample_grounded_task_call(
			task_name=selected_task_names[0],
			task_type_map=task_type_map,
			typed_objects=typed_objects,
		)
		repeated_atoms = apply_task_event_occurrence_suffixes((repeated_atom, repeated_atom))
		domain_label = self._few_shot_domain_label(typed_objects)

		few_shot_examples = (
			(
				f'Example 1 input: "Using {domain_label}, complete the tasks '
				f'{unordered_atoms[0]} and {unordered_atoms[1]} in any order."',
				json.dumps(
					{
						"ltlf_formula": build_unordered_eventuality_formula(unordered_atoms),
					},
					ensure_ascii=False,
				),
			),
			(
				f'Example 2 input: "Using {domain_label}, complete the tasks '
				f"{', then '.join(ordered_atoms)}.\"",
				json.dumps(
					{
						"ltlf_formula": build_ordered_benchmark_formula(ordered_atoms),
					},
					ensure_ascii=False,
				),
			),
			(
				f'Example 3 input: "Using {domain_label}, complete the tasks '
				f'{repeated_atom}, then {repeated_atom}."',
				json.dumps(
					{
						"ltlf_formula": build_ordered_benchmark_formula(repeated_atoms),
					},
					ensure_ascii=False,
				),
			),
		)
		lines: list[str] = []
		for index, (input_text, output_text) in enumerate(few_shot_examples, start=1):
			lines.append(input_text)
			lines.append(f"Example {index} output:")
			lines.append(output_text)
		return tuple(lines)

	@staticmethod
	def _few_shot_task_family(task_name_set: set[str]) -> Tuple[str, ...]:
		families = (
			("get_soil_data", "get_rock_data", "get_image_data"),
			("deliver", "get_to", "load"),
			("do_observation", "activate_instrument", "auto_calibrate"),
			("do_clear", "do_put_on", "do_on_table"),
		)
		for family in families:
			matched = tuple(task_name for task_name in family if task_name in task_name_set)
			if len(matched) >= 2:
				return matched
		return ()

	@staticmethod
	def _few_shot_domain_label(typed_objects: Dict[str, str]) -> str:
		if not typed_objects:
			return "the provided grounded objects"
		sorted_objects = sorted(str(object_name).strip() for object_name in typed_objects if str(object_name).strip())
		if len(sorted_objects) == 1:
			return f"the grounded object {sorted_objects[0]}"
		if len(sorted_objects) == 2:
			return f"the grounded objects {sorted_objects[0]} and {sorted_objects[1]}"
		return (
			"the grounded objects "
			+ ", ".join(sorted_objects[:3])
			+ (", ..." if len(sorted_objects) > 3 else "")
		)

	@staticmethod
	def _sample_grounded_task_call(
		*,
		task_name: str,
		task_type_map: Dict[str, Tuple[str, ...]],
		typed_objects: Dict[str, str],
	) -> str:
		argument_types = tuple(task_type_map.get(str(task_name), ()))
		args: list[str] = []
		for index, type_name in enumerate(argument_types, start=1):
			candidate = next(
				(
					str(object_name).strip()
					for object_name, object_type in sorted(typed_objects.items())
					if str(object_type).strip() == str(type_name).strip()
				),
				"",
			)
			args.append(candidate or f"{str(type_name).strip() or 'arg'}{index}")
		return f"{str(task_name).strip()}({', '.join(args)})"

	@staticmethod
	def _object_inventory_lines(typed_objects: Dict[str, str]) -> str:
		grouped: Dict[str, list[str]] = defaultdict(list)
		for object_name, type_name in sorted(typed_objects.items()):
			grouped[str(type_name).strip() or "object"].append(str(object_name).strip())
		lines = []
		for type_name, objects in grouped.items():
			lines.append(f"- {type_name}: {', '.join(objects)}")
		return "\n".join(lines)

	@classmethod
	def _prompt_object_inventory_lines(
		cls,
		query_text: str,
		typed_objects: Dict[str, str],
		*,
		max_listed_objects: int = 80,
	) -> str:
		"""
		Render a compact object inventory for the prompt without changing validation.

		Large benchmark problems can contain hundreds of objects even when the natural
		language query mentions only a small subset. Listing the full object set makes
		the grounding prompt unnecessarily long and has caused provider timeouts. The
		validator still receives the complete ``typed_objects`` dictionary.
		"""

		if len(typed_objects) <= max_listed_objects:
			return cls._object_inventory_lines(typed_objects)

		query_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_-]*", str(query_text or "")))
		referenced_objects = {
			object_name: type_name
			for object_name, type_name in typed_objects.items()
			if str(object_name).strip() in query_tokens
		}
		if not referenced_objects:
			group_counts: Dict[str, int] = defaultdict(int)
			for type_name in typed_objects.values():
				group_counts[str(type_name).strip() or "object"] += 1
			lines = [
				f"- {type_name}: {count} total objects available; none are explicitly named in the query text"
				for type_name, count in sorted(group_counts.items())
			]
			lines.append(
				f"- omitted_objects: {len(typed_objects)} objects are available for validation but not listed",
			)
			return "\n".join(lines)

		rendered = cls._object_inventory_lines(referenced_objects)
		omitted_count = max(0, len(typed_objects) - len(referenced_objects))
		if omitted_count:
			rendered = "\n".join(
				[
					rendered,
					f"- omitted_objects: {omitted_count} additional objects are available for validation but not listed because they are not referenced by the query text",
				],
			)
		return rendered

	def _validate_payload(
		self,
		*,
		query_text: str,
		payload: Dict[str, Any],
		method_library: Optional[HTNMethodLibrary],
		typed_objects: Dict[str, str],
		task_type_map: Dict[str, Tuple[str, ...]],
		type_parent_map: Dict[str, Optional[str]],
	) -> TemporalGroundingResult:
		ltlf_formula = self._normalise_ltlf_formula(str(payload.get("ltlf_formula") or ""))
		if not ltlf_formula:
			raise ValueError('Goal grounding response omitted required field "ltlf_formula".')
		if "{" in ltlf_formula or "}" in ltlf_formula:
			raise ValueError("LTLf formula string may not contain JSON braces.")
		unexpected_keys = sorted(
			key
			for key in payload.keys()
			if str(key).strip() and str(key).strip() != "ltlf_formula"
		)
		if unexpected_keys:
			raise ValueError(
				"Goal grounding response may contain only the key ltlf_formula. "
				"Unexpected keys: " + ", ".join(str(key) for key in unexpected_keys),
			)

		task_name_map = self._task_name_map(method_library)
		formula_atoms = self._extract_formula_atoms_in_order(ltlf_formula)
		if not formula_atoms:
			raise ValueError("LTLf formula does not contain any grounded task-event atoms.")

		seen_symbols: set[str] = set()
		subgoals = []
		for atom_expression in formula_atoms:
			raw_task_name, raw_args = self.symbol_normalizer.parse_predicate_string(atom_expression)
			event_task_name, raw_base_task_name, has_explicit_event_identity = (
				self._parse_task_event_predicate_name(raw_task_name)
			)
			task_name = task_name_map.get(raw_base_task_name, raw_base_task_name)
			if method_library is not None and method_library.task_for_name(task_name) is None:
				raise ValueError(
					f'LTLf formula references unknown grounded task "{raw_base_task_name}".',
				)
			args = tuple(str(arg).strip() for arg in raw_args if str(arg).strip())
			if any(arg.startswith("?") for arg in args):
				raise ValueError(
					f'LTLf formula contains lifted variables in task atom "{atom_expression}".',
				)
			argument_types = tuple(task_type_map.get(task_name, ()))
			if argument_types and len(args) != len(argument_types):
				raise ValueError(
					f'Task atom "{atom_expression}" has arity {len(args)} but task "{task_name}" '
					f'expects {len(argument_types)}.',
				)
			for index, arg in enumerate(args):
				if typed_objects and arg not in typed_objects:
					raise ValueError(
						f'Task atom "{atom_expression}" references unknown problem object "{arg}".',
					)
				if index >= len(argument_types) or arg not in typed_objects:
					continue
				actual_type = str(typed_objects[arg]).strip()
				expected_type = str(argument_types[index]).strip()
				if expected_type and not self._is_type_compatible(
					actual_type=actual_type,
					expected_type=expected_type,
					type_parent_map=type_parent_map,
				):
					raise ValueError(
						f'Task atom "{atom_expression}" argument "{arg}" has type "{actual_type}", '
						f'expected "{expected_type}".',
					)
			if not has_explicit_event_identity:
				event_task_name = raw_base_task_name
			symbol = self.symbol_normalizer.create_propositional_symbol(event_task_name, list(args))
			if symbol in seen_symbols:
				continue
			seen_symbols.add(symbol)
			subgoals.append(
				GroundedSubgoal(
					subgoal_id=symbol,
					task_name=task_name,
					args=args,
					argument_types=argument_types,
				),
			)

		return TemporalGroundingResult(
			query_text=query_text,
			ltlf_formula=ltlf_formula,
			subgoals=tuple(subgoals),
			typed_objects=dict(typed_objects),
			query_object_inventory=self._build_query_object_inventory(typed_objects),
			diagnostics=(),
		)

	@classmethod
	def _extract_formula_atoms(cls, ltlf_formula: str) -> set[str]:
		return set(cls._extract_formula_atoms_in_order(ltlf_formula))

	@classmethod
	def _extract_formula_atoms_in_order(cls, ltlf_formula: str) -> Tuple[str, ...]:
		ordered_atoms: list[str] = []
		invalid_tokens: set[str] = set()
		text = str(ltlf_formula or "")
		index = 0
		while index < len(text):
			if not (text[index].isalpha() or text[index] == "_"):
				index += 1
				continue
			start = index
			index += 1
			while index < len(text) and (text[index].isalnum() or text[index] == "_"):
				index += 1
			token = text[start:index]
			if token in cls._RESERVED_FORMULA_TOKENS or cls._FORMULA_OPERATOR_SEQUENCE_PATTERN.fullmatch(token):
				continue
			if index < len(text) and text[index] == "(":
				if token in cls._RESERVED_FORMULA_TOKENS or cls._FORMULA_OPERATOR_SEQUENCE_PATTERN.fullmatch(token):
					index += 1
					continue
				depth = 1
				index += 1
				while index < len(text) and depth > 0:
					if text[index] == "(":
						depth += 1
					elif text[index] == ")":
						depth -= 1
					index += 1
				token = text[start:index]
			if token.startswith("subgoal_") or token.startswith("query_step_"):
				invalid_tokens.add(token)
				continue
			ordered_atoms.append(token)
		if invalid_tokens:
			raise ValueError(
				"LTLf formula may not use placeholder atoms such as subgoal_* or query_step_*. Found: "
				+ ", ".join(sorted(invalid_tokens)),
			)
		return tuple(ordered_atoms)

	@classmethod
	def _parse_task_event_predicate_name(cls, raw_predicate_name: str) -> Tuple[str, str, bool]:
		predicate_name = str(raw_predicate_name or "").strip()
		match = cls._TASK_EVENT_SUFFIX_PATTERN.fullmatch(predicate_name)
		if match is None:
			return predicate_name, predicate_name, False
		return predicate_name, str(match.group("base") or "").strip(), True

	@staticmethod
	def _task_name_map(method_library: Optional[HTNMethodLibrary]) -> Dict[str, str]:
		task_name_map: Dict[str, str] = {}
		if method_library is None:
			return task_name_map
		for task in tuple(method_library.compound_tasks or ()) + tuple(method_library.primitive_tasks or ()):
			task_name = str(getattr(task, "name", "") or "").strip()
			source_name = str(getattr(task, "source_name", "") or "").strip()
			if task_name:
				task_name_map[task_name] = task_name
			if source_name:
				task_name_map[source_name] = task_name or source_name
		return task_name_map

	@staticmethod
	def _build_query_object_inventory(
		typed_objects: Dict[str, str],
	) -> Tuple[Dict[str, Any], ...]:
		grouped: Dict[str, list[str]] = defaultdict(list)
		for object_name, type_name in sorted(typed_objects.items()):
			grouped[str(type_name).strip() or "object"].append(str(object_name).strip())
		return tuple(
			{
				"type": type_name,
				"label": type_name,
				"objects": sorted(object_names),
			}
			for type_name, object_names in sorted(grouped.items())
		)

	@staticmethod
	def _is_type_compatible(
		*,
		actual_type: str,
		expected_type: str,
		type_parent_map: Dict[str, Optional[str]],
	) -> bool:
		if not actual_type or not expected_type:
			return True
		if actual_type == expected_type:
			return True
		cursor = actual_type
		visited: set[str] = set()
		while cursor and cursor not in visited:
			visited.add(cursor)
			cursor = str(type_parent_map.get(cursor) or "").strip() or None
			if cursor == expected_type:
				return True
		return False

	@staticmethod
	def _strip_code_fences(text: str) -> str:
		candidate = str(text or "").strip()
		if not candidate.startswith("```"):
			return candidate
		first_newline = candidate.find("\n")
		closing = candidate.rfind("```")
		if first_newline == -1 or closing <= first_newline:
			return candidate
		return candidate[first_newline + 1 : closing].strip()

	def _parse_json_blob(self, response_text: str) -> Dict[str, Any]:
		cleaned = self._strip_code_fences(response_text)
		try:
			payload = json.loads(cleaned)
		except json.JSONDecodeError:
			start_index = cleaned.find("{")
			end_index = cleaned.rfind("}")
			if start_index == -1 or end_index <= start_index:
				raise
			payload = json.loads(cleaned[start_index : end_index + 1])
		if not isinstance(payload, dict):
			raise ValueError("Goal grounding response must be one JSON object.")
		return payload

	def _create_chat_completion(
		self,
		messages: list[dict[str, str]],
		*,
		response_max_tokens: Optional[int] = None,
		request_timeout: Optional[float] = None,
		request_profile: Optional[Dict[str, Any]] = None,
	):
		profile = dict(request_profile or self._goal_grounding_request_profile(messages=messages))
		stream_response = bool(profile.get("stream_response", False))
		capped_response_max_tokens = self._apply_goal_grounding_provider_token_ceiling(
			response_max_tokens,
		)
		return create_openai_compatible_json_completion(
			self.client,
			model=self.model,
			messages=messages,
			timeout=float(request_timeout or self.request_timeout),
			max_tokens=capped_response_max_tokens,
			stream=stream_response,
			reasoning_effort=profile.get("reasoning_effort", "high"),
			thinking_type=profile.get("thinking_type", "enabled"),
		)

	def _read_response_payload(
		self,
		response: object,
		*,
		request_timeout: float,
		transport_metadata: Optional[Dict[str, Any]] = None,
	) -> Tuple[str, str, Dict[str, Any]]:
		metadata = dict(transport_metadata or {})
		finish_reason = self._extract_finish_reason(response)
		response_text = self._extract_response_text(response)
		metadata["llm_response_mode"] = "non_streaming"
		metadata["llm_finish_reason"] = finish_reason
		request_id = self._extract_transport_request_id(response)
		if request_id:
			metadata["llm_request_id"] = request_id
		return response_text, finish_reason, metadata

	def _extract_response_text(self, response: object) -> str:
		choices = getattr(response, "choices", None) or ()
		if not choices:
			raise GoalGroundingEmptyResponseError("LLM response did not include any choices.")

		message = getattr(choices[0], "message", None)
		if message is None:
			raise GoalGroundingEmptyResponseError(
				"LLM response choice did not include a message payload.",
			)

		for candidate in (
			getattr(message, "content", None),
			getattr(message, "parsed", None),
		):
			extracted = self._normalise_response_content(candidate)
			if extracted is not None:
				return extracted

		dumped_message = message.model_dump() if hasattr(message, "model_dump") else None
		if isinstance(dumped_message, dict):
			for key in ("content", "parsed", "output_text", "text"):
				extracted = self._normalise_response_content(dumped_message.get(key))
				if extracted is not None:
					return extracted
		response_dump = response.model_dump() if hasattr(response, "model_dump") else None
		if isinstance(response_dump, dict):
			extracted = self._extract_response_text_from_response_dump(response_dump)
			if extracted is not None:
				return extracted
		raise GoalGroundingEmptyResponseError(
			"LLM response did not contain any textual completion content.",
		)

	@staticmethod
	def _extract_finish_reason(response: object) -> str:
		choices = getattr(response, "choices", None) or ()
		if not choices:
			return ""
		return str(getattr(choices[0], "finish_reason", "") or "").strip()

	@staticmethod
	def _serialise_raw_response(response: object) -> str:
		if hasattr(response, "model_dump"):
			try:
				return json.dumps(response.model_dump(), ensure_ascii=False)
			except Exception:
				pass
		return repr(response)

	def _goal_grounding_request_profile(
		self,
		*,
		messages: Optional[list[dict[str, str]]] = None,
	) -> Dict[str, Any]:
		_ = messages
		return {
			"name": OPENAI_COMPATIBLE_JSON_PROFILE_NAME,
			"stream_response": False,
			"first_chunk_timeout_seconds": 0.0,
			"completion_max_tokens": max(int(self.response_max_tokens or 0), 1),
			"max_tokens_policy": "configured_ltlf_generation_max_tokens",
			"thinking_type": "enabled",
			"reasoning_effort": "high",
		}

	def _goal_grounding_transport_metadata(
		self,
		request_profile: Optional[Dict[str, Any]],
	) -> Dict[str, Any]:
		profile = dict(request_profile or {})
		metadata: Dict[str, Any] = {
			"llm_request_profile": profile.get("name"),
			"llm_first_chunk_timeout_seconds": profile.get("first_chunk_timeout_seconds"),
		}
		for metadata_key, profile_key in (
			("llm_completion_max_tokens", "completion_max_tokens"),
			("llm_max_tokens_policy", "max_tokens_policy"),
			("llm_thinking_type", "thinking_type"),
			("llm_reasoning_effort", "reasoning_effort"),
		):
			if profile.get(profile_key) is not None:
				metadata[metadata_key] = profile.get(profile_key)
		return metadata

	def _apply_goal_grounding_provider_token_ceiling(
		self,
		requested_max_tokens: Optional[int],
	) -> int | None:
		requested = max(int(requested_max_tokens or self.response_max_tokens or 0), 1)
		return requested

	@staticmethod
	def _run_with_wall_clock_timeout(
		timeout_seconds: Optional[float],
		callback,
	):
		effective_timeout_seconds = float(timeout_seconds or 0.0)
		if effective_timeout_seconds <= 0.0:
			return callback()
		if threading.current_thread() is not threading.main_thread():
			return callback()
		if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
			return callback()

		def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
			_ = (signum, frame)
			raise TimeoutError(
				"Goal-grounding LLM request exceeded the configured wall-clock "
				"timeout before a response chunk was created.",
			)

		previous_handler = signal.getsignal(signal.SIGALRM)
		try:
			signal.signal(signal.SIGALRM, _timeout_handler)
			signal.setitimer(signal.ITIMER_REAL, effective_timeout_seconds)
			return callback()
		finally:
			signal.setitimer(signal.ITIMER_REAL, 0.0)
			signal.signal(signal.SIGALRM, previous_handler)

	def _consume_streaming_llm_response(
		self,
		response: object,
		*,
		transport_metadata: Optional[Dict[str, Any]] = None,
		total_timeout_seconds: float = 0.0,
	) -> Tuple[str, str, Dict[str, Any]]:
		transport_metadata = dict(transport_metadata or {})
		transport_metadata["llm_response_mode"] = "streaming"
		request_id = self._extract_transport_request_id(response)
		if request_id:
			transport_metadata["llm_request_id"] = request_id
		handshake_seconds = getattr(response, "handshake_seconds", None)
		if handshake_seconds is not None:
			transport_metadata["llm_stream_handshake_seconds"] = handshake_seconds
		parts: list[str] = []
		reasoning_chunks_ignored = 0
		complete_payload: str | None = None
		finish_reason = ""
		close_stream = getattr(response, "close", None)
		deadline_expired = threading.Event()
		deadline_timer: Optional[threading.Timer] = None

		def _timeout_error() -> TimeoutError:
			error = TimeoutError(
				"Goal-grounding LLM call exceeded the configured timeout before "
				"returning a usable response.",
			)
			try:
				setattr(error, "transport_metadata", dict(transport_metadata))
			except Exception:
				pass
			return error

		def _close_stream_after_deadline() -> None:
			deadline_expired.set()
			if callable(close_stream):
				try:
					close_stream()
				except Exception:
					pass

		stream_start = time.monotonic()
		first_stream_chunk_recorded = False
		first_content_chunk_recorded = False
		first_chunk_timeout_seconds = float(
			transport_metadata.get("llm_first_chunk_timeout_seconds") or 0.0,
		)
		if total_timeout_seconds > 0.0:
			deadline_timer = threading.Timer(
				total_timeout_seconds,
				_close_stream_after_deadline,
			)
			deadline_timer.daemon = True
			deadline_timer.start()
		response_iterator = iter(response)
		try:
			while True:
				elapsed_seconds = time.monotonic() - stream_start
				if deadline_expired.is_set():
					raise _timeout_error()
				remaining_total_timeout_seconds = (
					total_timeout_seconds - elapsed_seconds
					if total_timeout_seconds > 0.0
					else 0.0
				)
				next_timeout_seconds: Optional[float] = None
				if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0:
					next_timeout_seconds = first_chunk_timeout_seconds - elapsed_seconds
					if total_timeout_seconds > 0.0:
						next_timeout_seconds = min(
							next_timeout_seconds,
							remaining_total_timeout_seconds,
						)
				elif total_timeout_seconds > 0.0:
					next_timeout_seconds = remaining_total_timeout_seconds
				if next_timeout_seconds is not None and next_timeout_seconds <= 0.0:
					timeout_error = TimeoutError(
						"Goal-grounding LLM call exceeded the configured first-chunk "
						"deadline before any streaming chunk arrived."
						if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0
						else "Goal-grounding LLM call exceeded the configured timeout "
						"before returning a usable response.",
					)
					try:
						setattr(timeout_error, "transport_metadata", dict(transport_metadata))
					except Exception:
						pass
					raise timeout_error
				try:
					chunk = self._run_with_wall_clock_timeout(
						next_timeout_seconds,
						lambda: next(response_iterator),
					)
				except StopIteration:
					if deadline_expired.is_set():
						raise _timeout_error()
					break
				except TimeoutError as exc:
					timeout_error = TimeoutError(
						"Goal-grounding LLM call exceeded the configured first-chunk "
						"deadline before any streaming chunk arrived."
						if not first_stream_chunk_recorded and first_chunk_timeout_seconds > 0.0
						else "Goal-grounding LLM call exceeded the configured timeout "
						"before returning a usable response.",
					)
					try:
						setattr(timeout_error, "transport_metadata", dict(transport_metadata))
					except Exception:
						pass
					raise timeout_error from exc
				except Exception as exc:
					if deadline_expired.is_set():
						raise _timeout_error() from exc
					raise
				request_id = self._extract_transport_request_id(chunk)
				if request_id:
					transport_metadata["llm_request_id"] = request_id
				if not first_stream_chunk_recorded:
					first_stream_chunk_seconds = round(time.monotonic() - stream_start, 6)
					transport_metadata["llm_first_stream_chunk_seconds"] = (
						first_stream_chunk_seconds
					)
					transport_metadata["llm_first_chunk_seconds"] = first_stream_chunk_seconds
					first_stream_chunk_recorded = True
				choices = getattr(chunk, "choices", None) or ()
				if not choices:
					continue
				choice = choices[0]
				finish_reason = str(getattr(choice, "finish_reason", "") or finish_reason).strip()
				delta = getattr(choice, "delta", None)
				for candidate in (
					getattr(delta, "content", None) if delta is not None else None,
					getattr(delta, "parsed", None) if delta is not None else None,
					getattr(choice, "message", None),
				):
					extracted = self._normalise_response_content(candidate)
					if extracted is None:
						continue
					if not first_content_chunk_recorded:
						transport_metadata["llm_first_content_chunk_seconds"] = round(
							time.monotonic() - stream_start,
							6,
						)
						first_content_chunk_recorded = True
					parts.append(extracted)
				for reasoning_candidate in (
					getattr(delta, "reasoning", None) if delta is not None else None,
					getattr(delta, "reasoning_content", None) if delta is not None else None,
					getattr(delta, "reasoning_details", None) if delta is not None else None,
					getattr(choice, "reasoning", None),
				):
					if reasoning_candidate is None:
						continue
					reasoning_chunks_ignored += 1
					transport_metadata["llm_reasoning_chunks_ignored"] = reasoning_chunks_ignored
					if "llm_first_reasoning_chunk_seconds" not in transport_metadata:
						transport_metadata["llm_first_reasoning_chunk_seconds"] = round(
							time.monotonic() - stream_start,
							6,
						)
				current_text = "".join(parts).strip()
				candidate_payload = self._extract_complete_json_payload_text(current_text)
				if candidate_payload is not None and complete_payload is None:
					transport_metadata["llm_complete_json_seconds"] = round(
						time.monotonic() - stream_start,
						6,
					)
					complete_payload = candidate_payload
				if total_timeout_seconds > 0.0 and (time.monotonic() - stream_start) >= total_timeout_seconds:
					raise _timeout_error()
			text = "".join(parts).strip()
			if complete_payload is None:
				complete_payload = self._extract_complete_json_payload_text(text)
				if complete_payload is not None:
					transport_metadata["llm_complete_json_seconds"] = round(
						time.monotonic() - stream_start,
						6,
					)
			transport_metadata["llm_finish_reason"] = finish_reason or "stop"
			if finish_reason == "length":
				error = RuntimeError(
					"LLM response was truncated before completion (finish_reason=length).",
				)
				try:
					setattr(error, "transport_metadata", dict(transport_metadata))
				except Exception:
					pass
				raise error
			if complete_payload is not None:
				return complete_payload, finish_reason or "stop", transport_metadata
			if text:
				error = GoalGroundingMalformedResponseError(
					"LLM response text did not contain a complete usable JSON object. "
					f"finish_reason={finish_reason!r}",
				)
				try:
					setattr(error, "transport_metadata", dict(transport_metadata))
				except Exception:
					pass
				raise error
			error = GoalGroundingEmptyResponseError(
				"LLM response returned empty textual completion content. "
				f"finish_reason={finish_reason!r}",
			)
			try:
				setattr(error, "transport_metadata", dict(transport_metadata))
			except Exception:
				pass
			raise error
		finally:
			if deadline_timer is not None:
				deadline_timer.cancel()
			if callable(close_stream):
				close_stream()

	@staticmethod
	def _normalise_response_content(content: object) -> str | None:
		if content is None:
			return None
		if isinstance(content, str):
			text = content.strip()
			return text or None
		if isinstance(content, dict):
			for key in ("text", "value", "content"):
				extracted = NLToLTLfGenerator._normalise_response_content(content.get(key))
				if extracted is not None:
					return extracted
			return json.dumps(content, ensure_ascii=False)
		if isinstance(content, (list, tuple)):
			parts = [
				extracted
				for item in content
				if (extracted := NLToLTLfGenerator._normalise_response_content(item)) is not None
			]
			return "\n".join(parts).strip() or None
		text_attr = getattr(content, "text", None)
		extracted = NLToLTLfGenerator._normalise_response_content(text_attr)
		if extracted is not None:
			return extracted
		value_attr = getattr(content, "value", None)
		extracted = NLToLTLfGenerator._normalise_response_content(value_attr)
		if extracted is not None:
			return extracted
		stringified = str(content).strip()
		return stringified or None

	@classmethod
	def _extract_response_text_from_response_dump(cls, response_dump: Dict[str, Any]) -> str | None:
		choices = response_dump.get("choices")
		if isinstance(choices, list) and choices:
			first_choice = choices[0]
			if isinstance(first_choice, dict):
				message = first_choice.get("message")
				if isinstance(message, dict):
					for key in ("content", "parsed", "output_text", "text"):
						extracted = cls._normalise_response_content(message.get(key))
						if extracted is not None:
							return extracted
				for key in ("content", "parsed", "output_text", "text"):
					extracted = cls._normalise_response_content(first_choice.get(key))
					if extracted is not None:
						return extracted
		for key in ("output_text", "text", "content", "parsed"):
			extracted = cls._normalise_response_content(response_dump.get(key))
			if extracted is not None:
				return extracted
		return None

	@staticmethod
	def _extract_transport_request_id(payload: object) -> str | None:
		for attr_name in ("id", "response_id", "request_id", "_request_id"):
			value = getattr(payload, attr_name, None)
			if isinstance(value, str) and value.strip():
				return value.strip()
		response_payload = getattr(payload, "response", None)
		if response_payload is not None:
			for headers_attr in ("headers", "_headers"):
				headers = getattr(response_payload, headers_attr, None)
				if headers is None:
					continue
				for key in ("x-request-id", "request-id", "openai-request-id"):
					try:
						value = headers.get(key)
					except Exception:
						value = None
					if isinstance(value, str) and value.strip():
						return value.strip()
		return None

	@staticmethod
	def _appears_truncated_json(text: str) -> bool:
		open_curly = text.count("{")
		close_curly = text.count("}")
		open_square = text.count("[")
		close_square = text.count("]")
		if open_curly > close_curly:
			return True
		if open_square > close_square:
			return True
		return False

	@classmethod
	def _extract_complete_json_payload_text(cls, text: str) -> str | None:
		stripped = str(text or "").strip()
		if not stripped:
			return None

		def _decode_from(start_index: int) -> str | None:
			candidate = stripped[start_index:]
			if cls._appears_truncated_json(candidate):
				return None
			try:
				parsed, end_index = json.JSONDecoder().raw_decode(candidate)
			except json.JSONDecodeError:
				return None
			if not isinstance(parsed, (dict, list)):
				return None
			payload = candidate[:end_index].strip()
			return payload or None

		first_nonspace = stripped[0]
		if first_nonspace in "{[":
			payload = _decode_from(0)
			if payload is not None:
				return payload
		object_index = stripped.find("{")
		if object_index != -1:
			payload = _decode_from(object_index)
			if payload is not None:
				return payload
		array_index = stripped.find("[")
		if array_index != -1:
			payload = _decode_from(array_index)
			if payload is not None:
				return payload
		return None

	@staticmethod
	def _suggest_response_max_tokens(query_text: str) -> int:
		query_length = len(str(query_text or ""))
		task_call_mentions = len(
			re.findall(r"\b[a-z_][a-z0-9_]*\([^()]*\)", str(query_text or "")),
		)
		if task_call_mentions >= 40:
			return 64000
		if task_call_mentions >= 20:
			return 48000
		if task_call_mentions >= 12:
			return 32000
		if task_call_mentions >= 8:
			return 24000
		if query_length >= 20000:
			return 64000
		if query_length >= 10000:
			return 48000
		if query_length >= 6000:
			return 40000
		if query_length >= 3000:
			return 32000
		if query_length >= 1200:
			return 20000
		return 12000

	@staticmethod
	def _suggest_request_timeout(query_text: str) -> float:
		query_length = len(str(query_text or ""))
		task_call_mentions = len(
			re.findall(r"\b[a-z_][a-z0-9_]*\([^()]*\)", str(query_text or "")),
		)
		if task_call_mentions >= 30 or query_length >= 30000:
			return 300.0
		if task_call_mentions >= 18 or query_length >= 18000:
			return 240.0
		if task_call_mentions >= 10 or query_length >= 9000:
			return 180.0
		if task_call_mentions >= 4 or query_length >= 3000:
			return 120.0
		return 60.0

	@staticmethod
	def _normalise_ltlf_formula(ltlf_formula: str) -> str:
		text = re.sub(r"\s+", " ", str(ltlf_formula or "").strip())
		if not text:
			return ""
		normalised_chars: list[str] = []
		open_parentheses = 0
		for character in text:
			if character == "(":
				open_parentheses += 1
				normalised_chars.append(character)
				continue
			if character == ")":
				if open_parentheses <= 0:
					continue
				open_parentheses -= 1
				normalised_chars.append(character)
				continue
			normalised_chars.append(character)
		if open_parentheses > 0:
			normalised_chars.extend(")" for _ in range(open_parentheses))
		return "".join(normalised_chars).strip()
