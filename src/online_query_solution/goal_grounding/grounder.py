"""
LLM-driven temporal goal grounding for the Jason online runtime.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from offline_method_generation.method_synthesis.schema import HTNMethodLibrary
from online_query_solution.artifacts import GroundedSubgoal, TemporalGroundingResult
from utils.config import DEFAULT_GOAL_GROUNDING_MODEL
from utils.symbol_normalizer import SymbolNormalizer


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
	) -> None:
		self.api_key = api_key
		self.model = model or DEFAULT_GOAL_GROUNDING_MODEL
		self.base_url = base_url
		self.domain_file = domain_file
		self.request_timeout = float(request_timeout or 60.0)
		self.response_max_tokens = int(response_max_tokens or 12000)
		self.client = None
		self.last_generation_metadata: Dict[str, Any] = {}
		self.symbol_normalizer = SymbolNormalizer()

		self.domain = None
		if domain_file:
			from utils.hddl_parser import HDDLParser

			self.domain = HDDLParser.parse_domain(domain_file)

		if api_key:
			from openai import OpenAI

			client_kwargs: Dict[str, Any] = {
				"api_key": api_key,
				"timeout": self.request_timeout,
				"max_retries": 0,
			}
			if base_url:
				client_kwargs["base_url"] = base_url
			self.client = OpenAI(**client_kwargs)

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
				"No API key configured. Please set OPENAI_API_KEY in .env file.",
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
		result = None
		attempt_count = 1
		try:
			response = self._create_chat_completion(
				[
					{"role": "system", "content": last_prompt["system"]},
					{"role": "user", "content": last_prompt["user"]},
				],
				response_max_tokens=int(current_attempt["response_max_tokens"]),
				request_timeout=float(current_attempt["request_timeout"]),
			)
			finish_reason = self._extract_finish_reason(response)
			last_finish_reason = finish_reason
			last_response_text = self._serialise_raw_response(response)
			response_text = self._extract_response_text(response)
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
				"attempt_errors": attempt_errors,
				"task_event_count": len(result.subgoals),
				"formula_atom_count": len(self._extract_formula_atoms(result.ltlf_formula)),
				"model": self.model,
				"last_prompt": dict(last_prompt),
				"last_response": response_text,
				"last_finish_reason": finish_reason,
			}
			return result, last_prompt, response_text
		except Exception as exc:
			last_error = exc
			attempt_errors.append(
				{
					"mode": str(current_attempt["mode"]),
					"error": str(exc),
					"finish_reason": last_finish_reason,
				},
			)
			self.last_generation_metadata = {
				"attempt_mode": str(current_attempt["mode"]),
				"attempt_count": attempt_count,
				"attempt_errors": list(attempt_errors),
				"model": self.model,
				"last_prompt": dict(last_prompt),
				"last_response": last_response_text,
				"last_finish_reason": last_finish_reason,
			}
		if last_error is None:
			raise RuntimeError("Goal grounding failed without a recorded exception.")
		self.last_generation_metadata = {
			"attempt_mode": str(current_attempt["mode"]),
			"attempt_count": attempt_count,
			"attempt_errors": attempt_errors,
			"model": self.model,
			"last_prompt": dict(last_prompt),
			"last_response": last_response_text,
			"last_response_preview": last_response_text[:500],
			"last_finish_reason": last_finish_reason,
		}
		raise last_error

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
		object_lines = self._object_inventory_lines(typed_objects)
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
				"Callable grounded online tasks:",
				task_lines or "- none",
				"",
				"Available grounded problem objects:",
				object_lines or "- none",
				"",
				"Output contract:",
				'- Return exactly one JSON object with the single key "ltlf_formula".',
				'- "ltlf_formula" must be a single LTLf formula string.',
				'- Output must be minified JSON with no markdown fences, no prose, and no extra keys.',
				"- Every atomic proposition in ltlf_formula must itself be one grounded task-event call.",
				"- Use atoms in task-call form such as do_put_on(b4, b2) or do_clear(b2).",
				"- If the same grounded task call must occur multiple times, each repeated occurrence must use a distinct grounded task-event identity by appending an occurrence suffix to the task name, such as do_put_on__e1(b8, b9) and do_put_on__e2(b8, b9).",
				"- Never reuse the exact same task-event atom twice in one formula.",
				"- Do not use placeholder atoms such as subgoal_1, query_step_1, goal_1, or event_1.",
				'- Do not output lifted variables such as ?x. All args must be grounded problem objects from the provided inventory.',
				'- Do not invent tasks or objects.',
				"- If the query begins with an object inventory such as 'Using blocks ...' or any other setup inventory, treat that inventory as context only, not as task-event atoms.",
				"- Create task-event atoms only from explicitly requested grounded task invocations in the query.",
				"- Do not create task-event atoms from bare object names, setup inventories, or explanatory text.",
				"- If the query literally lists grounded task calls after a phrase like 'complete the tasks', copy exactly those task calls into the formula and keep their order.",
				"- If that listed task sequence repeats the same grounded task call multiple times, preserve the repeated order and count, but distinguish each repeated occurrence with a unique __eN suffix on the task name.",
				"- Treat a comma-separated task list after 'complete the tasks' as ordered by default. Preserve the listed order in ltlf_formula unless the query explicitly states that order does not matter or may be arbitrary.",
				"- If the query is ambiguous, still return your best grounded interpretation as one formula. Do not add extra fields.",
				"",
				"Formula syntax reminders:",
				"- Atoms: grounded task-event calls such as do_put_on(b4, b2)",
				"- Repeated grounded task calls must use occurrence-tagged atoms such as do_put_on__e1(b8, b9) and do_put_on__e2(b8, b9).",
				"- Unary operators: F, G, X, WX",
				"- Binary operators: U, R",
				"- Boolean operators: !, &, |, ->, <->",
				"- Constants: true, false",
				"- For an explicit ordered sequence 'A then B then C', prefer a right-nested eventuality such as F(do_a(x) & F(do_b(y) & F(do_c(z)))).",
				"- Use a plain conjunction of eventualities only when the query does not impose an order.",
				"",
				"Few-shot examples:",
				'Example 1 input: "Using blocks b1 and b2, complete the tasks do_clear(b2) and do_put_on(b1, b2) in any order."',
				"Example 1 output:",
				json.dumps(
					{
						"ltlf_formula": "F(do_clear(b2)) & F(do_put_on(b1, b2))",
					},
					ensure_ascii=False,
				),
				'Example 2 input: "Using blocks b4, b2, b1, and b3, complete the tasks do_put_on(b4, b2), then do_put_on(b1, b4), then do_put_on(b3, b1)."',
				"Example 2 output:",
				json.dumps(
					{
						"ltlf_formula": "F(do_put_on(b4, b2) & F(do_put_on(b1, b4) & F(do_put_on(b3, b1))))",
					},
					ensure_ascii=False,
				),
				'Example 3 input: "Using blocks b8 and b9, complete the tasks do_put_on(b8, b9), then do_put_on(b8, b9)."',
				"Example 3 output:",
				json.dumps(
					{
						"ltlf_formula": "F(do_put_on__e1(b8, b9) & F(do_put_on__e2(b8, b9)))",
					},
					ensure_ascii=False,
				),
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
		if method_library is None:
			return ""
		lines = []
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

	@staticmethod
	def _object_inventory_lines(typed_objects: Dict[str, str]) -> str:
		grouped: Dict[str, list[str]] = defaultdict(list)
		for object_name, type_name in sorted(typed_objects.items()):
			grouped[str(type_name).strip() or "object"].append(str(object_name).strip())
		lines = []
		for type_name, objects in grouped.items():
			lines.append(f"- {type_name}: {', '.join(objects)}")
		return "\n".join(lines)

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

		formula_atom_occurrences: Dict[str, int] = defaultdict(int)
		for atom_expression in formula_atoms:
			formula_atom_occurrences[str(atom_expression).strip()] += 1

		seen_symbols: set[str] = set()
		subgoals = []
		for atom_expression in formula_atoms:
			raw_task_name, raw_args = self.symbol_normalizer.parse_predicate_string(atom_expression)
			event_task_name, raw_base_task_name, has_explicit_event_identity = (
				self._parse_task_event_predicate_name(raw_task_name)
			)
			if formula_atom_occurrences[str(atom_expression).strip()] > 1:
				raise ValueError(
					"Repeated grounded task call requires explicit grounded task-event identities. "
					f'Use distinct atoms such as "{raw_base_task_name}__e1(...)" and '
					f'"{raw_base_task_name}__e2(...)" instead of repeating "{atom_expression}".',
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
				raise ValueError(
					f'LTLf formula reuses grounded task-event identity "{atom_expression}". '
					"Each event atom must be unique within the formula.",
				)
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
	):
		request_kwargs = {
			"model": self.model,
			"messages": messages,
			"temperature": 0.0,
			"timeout": float(request_timeout or self.request_timeout),
			"max_tokens": int(response_max_tokens or self.response_max_tokens),
			"response_format": {"type": "json_object"},
		}
		extra_body = self._openrouter_provider_routing_body()
		if extra_body is not None:
			request_kwargs["extra_body"] = extra_body
		return self.client.chat.completions.create(**request_kwargs)

	def _extract_response_text(self, response: object) -> str:
		choices = getattr(response, "choices", None) or ()
		if not choices:
			raise RuntimeError("LLM response did not include any choices.")

		message = getattr(choices[0], "message", None)
		if message is None:
			raise RuntimeError("LLM response choice did not include a message payload.")

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
		raise RuntimeError("LLM response did not contain usable textual JSON content.")

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

	def _openrouter_provider_routing_body(self) -> Dict[str, Any] | None:
		base_url = str(self.base_url or "").strip().lower()
		if "openrouter.ai" not in base_url:
			return None
		model_name = str(self.model or "").strip()
		if "/" not in model_name:
			return None
		provider_name = model_name.split("/", 1)[0].strip().lower()
		if not provider_name:
			return None
		extra_body: Dict[str, Any] = {
			"provider": {
				"only": [provider_name],
				"allow_fallbacks": False,
			},
		}
		if provider_name == "minimax":
			extra_body["reasoning"] = {
				"max_tokens": 1,
				"exclude": True,
			}
		return extra_body

	def _normalise_response_content(self, content: object) -> str | None:
		if content is None:
			return None
		if isinstance(content, str):
			text = content.strip()
			return text or None
		if isinstance(content, dict):
			for key in ("text", "value", "content"):
				extracted = self._normalise_response_content(content.get(key))
				if extracted is not None:
					return extracted
			return json.dumps(content, ensure_ascii=False)
		if isinstance(content, (list, tuple)):
			parts = [
				extracted
				for item in content
				if (extracted := self._normalise_response_content(item)) is not None
			]
			return "\n".join(parts).strip() or None
		text_attr = getattr(content, "text", None)
		extracted = self._normalise_response_content(text_attr)
		if extracted is not None:
			return extracted
		stringified = str(content).strip()
		return stringified or None

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
		if query_length >= 30000:
			return 180.0
		if query_length >= 20000:
			return 150.0
		if query_length >= 10000:
			return 120.0
		if query_length >= 6000:
			return 90.0
		if query_length >= 3000:
			return 60.0
		return 30.0

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
