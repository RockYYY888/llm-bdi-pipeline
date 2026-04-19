"""
LLM-driven temporal goal grounding for the Jason online runtime.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

from offline_method_generation.method_synthesis.schema import HTNMethodLibrary
from online_query_solution.artifacts import GroundedSubgoal, TemporalGroundingResult
from utils.config import DEFAULT_GOAL_GROUNDING_MODEL


class NLToLTLfGenerator:
	"""
	Convert a natural-language query into one grounded LTLf formula plus grounded subgoals.
	"""

	_COMPACT_PROMPT_QUERY_LENGTH_THRESHOLD = 1200
	_SUBGOAL_ID_PATTERN = re.compile(r"subgoal_\d+$")
	_FORMULA_TOKEN_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
	_FORMULA_OPERATOR_SEQUENCE_PATTERN = re.compile(r"^(?:WX|[FGXURW])+$")
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
		last_error: Exception | None = None
		last_response_text = ""
		last_prompt: Dict[str, str] = {}
		attempt_errors: list[Dict[str, str]] = []
		result = None
		for attempt in prompt_attempts:
			system_prompt = str(attempt["system"])
			user_prompt = str(attempt["user"])
			last_prompt = {"system": system_prompt, "user": user_prompt}
			response = self._create_chat_completion(
				[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				response_max_tokens=int(attempt["response_max_tokens"]),
			)
			response_text = self._extract_response_text(response)
			last_response_text = response_text
			try:
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
					"attempt_mode": str(attempt["mode"]),
					"attempt_count": int(attempt["index"]),
					"attempt_errors": attempt_errors,
					"subgoal_count": len(result.subgoals),
					"formula_atom_count": len(self._extract_formula_atoms(result.ltlf_formula)),
					"model": self.model,
				}
				return result, last_prompt, response_text
			except Exception as exc:
				last_error = exc
				attempt_errors.append(
					{
						"mode": str(attempt["mode"]),
						"error": str(exc),
					},
				)
				self.last_generation_metadata = {
					"attempt_mode": str(attempt["mode"]),
					"attempt_count": int(attempt["index"]),
					"attempt_errors": list(attempt_errors),
					"model": self.model,
					"last_prompt": dict(last_prompt),
					"last_response": response_text,
				}
				continue

		if last_error is None:
			raise RuntimeError("Goal grounding failed without a recorded exception.")
		self.last_generation_metadata = {
			"attempt_mode": str(prompt_attempts[-1]["mode"]),
			"attempt_count": len(prompt_attempts),
			"attempt_errors": attempt_errors,
			"model": self.model,
			"last_response_preview": last_response_text[:500],
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
		compact_system, compact_user = self._build_compact_prompts(
			query_text=query_text,
			method_library=method_library,
			typed_objects=typed_objects,
			task_type_map=task_type_map,
		)
		base_max_tokens = int(self.response_max_tokens)
		compact_max_tokens = max(
			base_max_tokens,
			self._suggest_response_max_tokens(query_text),
		)
		long_query = len(query_text) >= self._COMPACT_PROMPT_QUERY_LENGTH_THRESHOLD
		ordered_attempts = list(
			(
				[
					("compact", compact_system, compact_user, compact_max_tokens),
					("standard", standard_system, standard_user, base_max_tokens),
				]
				if long_query
				else [
					("standard", standard_system, standard_user, base_max_tokens),
					("compact", compact_system, compact_user, compact_max_tokens),
				]
			)
		)
		return tuple(
			{
				"index": index,
				"mode": mode,
				"system": system_prompt,
				"user": user_prompt,
				"response_max_tokens": response_max_tokens,
			}
			for index, (mode, system_prompt, user_prompt, response_max_tokens) in enumerate(
				ordered_attempts,
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
				"You must not explain your reasoning. Return JSON only.",
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
				'- Return exactly one JSON object with keys "ltlf_formula", "subgoals", and optional "diagnostics".',
				'- "ltlf_formula" must be a single LTLf formula string.',
				'- "subgoals" must be a list of grounded task calls with schema {"id": "subgoal_N", "task_name": "...", "args": ["..."]}.',
				'- Every atomic proposition appearing in ltlf_formula must be one of the declared subgoal ids.',
				'- Do not use query_step ids. Use subgoal ids only.',
				'- Do not output lifted variables such as ?x. All args must be grounded problem objects from the provided inventory.',
				'- Do not invent tasks or objects.',
				'- If the query is ambiguous, still return your best grounded interpretation and put the ambiguity note in diagnostics.',
				"",
				"Formula syntax reminders:",
				"- Atoms: subgoal_1, subgoal_2, ...",
				"- Unary operators: F, G, X, WX",
				"- Binary operators: U, R",
				"- Boolean operators: !, &, |, ->, <->",
				"- Constants: true, false",
				"- For an explicit ordered sequence 'A then B then C', prefer a right-nested eventuality such as F(subgoal_1 & F(subgoal_2 & F(subgoal_3))).",
				"- Use a plain conjunction of eventualities only when the query does not impose an order.",
				"",
				"Example output:",
				json.dumps(
					{
						"ltlf_formula": "F(subgoal_1) & F(subgoal_2)",
						"subgoals": [
							{"id": "subgoal_1", "task_name": "clear_block", "args": ["a"]},
							{"id": "subgoal_2", "task_name": "stack", "args": ["b", "a"]},
						],
						"diagnostics": [],
					},
					ensure_ascii=False,
				),
				"",
				"Ordered example:",
				json.dumps(
					{
						"ltlf_formula": "F(subgoal_1 & F(subgoal_2 & F(subgoal_3)))",
						"subgoals": [
							{"id": "subgoal_1", "task_name": "do_put_on", "args": ["b4", "b2"]},
							{"id": "subgoal_2", "task_name": "do_put_on", "args": ["b1", "b4"]},
							{"id": "subgoal_3", "task_name": "do_put_on", "args": ["b3", "b1"]},
						],
						"diagnostics": [],
					},
					ensure_ascii=False,
				),
			],
		)
		user_prompt = "\n".join(
			[
				"Ground this query into one LTLf formula plus grounded subgoals.",
				f'Query: "{query_text}"',
			],
		)
		return system_prompt, user_prompt

	def _build_compact_prompts(
		self,
		*,
		query_text: str,
		method_library: Optional[HTNMethodLibrary],
		typed_objects: Dict[str, str],
		task_type_map: Dict[str, Tuple[str, ...]],
	) -> Tuple[str, str]:
		task_lines = self._task_inventory_lines(method_library, task_type_map)
		object_lines = self._object_inventory_lines(typed_objects)
		system_prompt = "\n".join(
			[
				"Return minified JSON only.",
				"Convert the query into one grounded LTLf formula and grounded subgoals.",
				"Preserve temporal meaning exactly.",
				"If the query already names grounded task calls, copy those task calls exactly and keep their surface order.",
				'Use exactly the keys "ltlf_formula", "subgoals", and optional "diagnostics".',
				'"subgoals" must be a non-empty list and must contain one grounded entry for every task mention in the query.',
				'Use subgoal ids only: "subgoal_1", "subgoal_2", ...',
				"All args must be grounded problem objects from the provided inventory.",
				'For an ordered task list, use a right-nested formula like F(subgoal_1 & F(subgoal_2 & F(subgoal_3))).',
				'For an unordered task set, use a conjunction like F(subgoal_1) & F(subgoal_2) & F(subgoal_3).',
				"",
				"Callable tasks:",
				task_lines or "- none",
				"",
				"Available objects:",
				object_lines or "- none",
			],
		)
		user_prompt = "\n".join(
			[
				"Emit minified JSON with no markdown or commentary.",
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

		raw_subgoals = payload.get("subgoals")
		if not isinstance(raw_subgoals, list) or not raw_subgoals:
			raise ValueError('Goal grounding response must contain a non-empty "subgoals" list.')

		task_name_map = self._task_name_map(method_library)
		seen_subgoal_ids: set[str] = set()
		subgoals = []
		for raw_item in raw_subgoals:
			if not isinstance(raw_item, dict):
				raise ValueError("Every subgoal entry must be a JSON object.")
			subgoal_id = str(raw_item.get("id") or raw_item.get("subgoal_id") or "").strip()
			if not self._SUBGOAL_ID_PATTERN.fullmatch(subgoal_id):
				raise ValueError(
					f'Invalid subgoal id "{subgoal_id}". Expected ids matching "subgoal_N".',
				)
			if subgoal_id in seen_subgoal_ids:
				raise ValueError(f'Duplicate subgoal id "{subgoal_id}".')
			seen_subgoal_ids.add(subgoal_id)

			raw_task_name = str(raw_item.get("task_name") or "").strip()
			if not raw_task_name:
				raise ValueError(f'Subgoal "{subgoal_id}" omitted task_name.')
			task_name = task_name_map.get(raw_task_name, raw_task_name)
			if method_library is not None and method_library.task_for_name(task_name) is None:
				raise ValueError(
					f'Subgoal "{subgoal_id}" references unknown task "{raw_task_name}".',
				)

			raw_args = raw_item.get("args") or ()
			if not isinstance(raw_args, (list, tuple)):
				raise ValueError(f'Subgoal "{subgoal_id}" args must be a list.')
			args = tuple(str(arg).strip() for arg in raw_args if str(arg).strip())
			if any(arg.startswith("?") for arg in args):
				raise ValueError(
					f'Subgoal "{subgoal_id}" contains lifted variables: {list(args)}.',
				)
			argument_types = tuple(task_type_map.get(task_name, ()))
			if argument_types and len(args) != len(argument_types):
				raise ValueError(
					f'Subgoal "{subgoal_id}" arity mismatch for task "{task_name}": '
					f"{len(args)} vs {len(argument_types)}.",
				)
			for index, arg in enumerate(args):
				if typed_objects and arg not in typed_objects:
					raise ValueError(
						f'Subgoal "{subgoal_id}" references unknown problem object "{arg}".',
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
						f'Subgoal "{subgoal_id}" argument "{arg}" has type "{actual_type}", '
						f'expected "{expected_type}".',
					)
			subgoals.append(
				GroundedSubgoal(
					subgoal_id=subgoal_id,
					task_name=task_name,
					args=args,
					argument_types=argument_types,
				),
			)

		formula_atoms = self._extract_formula_atoms(ltlf_formula)
		declared_subgoal_ids = {subgoal.subgoal_id for subgoal in subgoals}
		undeclared_atoms = sorted(formula_atoms - declared_subgoal_ids)
		if undeclared_atoms:
			raise ValueError(
				"LTLf formula references undeclared subgoals: " + ", ".join(undeclared_atoms),
			)
		unused_subgoals = sorted(declared_subgoal_ids - formula_atoms)
		if unused_subgoals:
			raise ValueError(
				"Subgoals not referenced by ltlf_formula: " + ", ".join(unused_subgoals),
			)

		return TemporalGroundingResult(
			query_text=query_text,
			ltlf_formula=ltlf_formula,
			subgoals=tuple(subgoals),
			typed_objects=dict(typed_objects),
			query_object_inventory=self._build_query_object_inventory(typed_objects),
			diagnostics=tuple(
				str(item).strip()
				for item in (payload.get("diagnostics") or ())
				if str(item).strip()
			),
		)

	@classmethod
	def _extract_formula_atoms(cls, ltlf_formula: str) -> set[str]:
		atoms: set[str] = set()
		invalid_tokens: set[str] = set()
		for token in cls._FORMULA_TOKEN_PATTERN.findall(str(ltlf_formula or "")):
			if token in cls._RESERVED_FORMULA_TOKENS:
				continue
			if cls._FORMULA_OPERATOR_SEQUENCE_PATTERN.fullmatch(token):
				continue
			if cls._SUBGOAL_ID_PATTERN.fullmatch(token):
				atoms.add(token)
				continue
			invalid_tokens.add(token)
		if invalid_tokens:
			raise ValueError(
				"LTLf formula may reference only subgoal_* atoms. Found: "
				+ ", ".join(sorted(invalid_tokens)),
			)
		return atoms

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
	):
		request_kwargs = {
			"model": self.model,
			"messages": messages,
			"temperature": 0.0,
			"timeout": self.request_timeout,
			"max_tokens": int(response_max_tokens or self.response_max_tokens),
		}
		request_variants = [
			{"response_format": {"type": "json_object"}},
			{},
		]
		last_optional_error: Exception | None = None
		for variant in request_variants:
			try:
				return self.client.chat.completions.create(
					**request_kwargs,
					**variant,
				)
			except Exception as exc:
				if self._is_unsupported_json_response_format_error(exc):
					last_optional_error = exc
					continue
				raise
		if last_optional_error is not None:
			raise last_optional_error
		raise RuntimeError("goal grounding could not create a chat completion request.")

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
		if query_length >= 6000:
			return 32000
		if query_length >= 3000:
			return 24000
		if query_length >= 1200:
			return 16000
		return 12000

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

	@staticmethod
	def _is_unsupported_json_response_format_error(exc: Exception) -> bool:
		message = str(exc).lower()
		if "response_format" not in message and "json_object" not in message:
			return False
		return any(
			marker in message
			for marker in (
				"unsupported",
				"not supported",
				"invalid parameter",
				"unknown parameter",
				"unrecognized request argument",
				"extra inputs are not permitted",
			)
		)
