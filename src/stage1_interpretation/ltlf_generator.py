"""
LTLf Generator Module

This module converts natural language instructions into LTLf formulas using LLM.

**LTLf Syntax Reference**: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
"""

import json
import re
import time
from typing import Optional

from .ltlf_formula import LTLFormula, LTLSpecification, TemporalOperator, LogicalOperator
from .grounding_map import GroundingMap, create_propositional_symbol


class NLToLTLfGenerator:
    """
    Converts Natural Language to LTLf formulas using LLM

    Uses OpenAI API to understand natural language and generate
    structured LTLf specifications for temporal goals.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 domain_file: Optional[str] = None,
                 request_timeout: Optional[float] = None,
                 response_max_tokens: Optional[int] = None):
        """
        Initialize generator with optional API key, model, base URL, and domain file

        Args:
            api_key: OpenAI API key
            model: Model name
            base_url: Custom API base URL
            domain_file: Path to HDDL domain file (for dynamic prompt construction)
            request_timeout: Maximum per-request timeout in seconds
            response_max_tokens: Maximum completion tokens for the JSON response
        """
        self.api_key = api_key
        self.model = model or "deepseek-chat"
        self.base_url = base_url
        self.domain_file = domain_file
        self.request_timeout = float(request_timeout or 60.0)
        self.response_max_tokens = int(response_max_tokens or 12000)
        self.client = None
        self.last_generation_metadata: dict = {}

        # Parse domain if provided
        self.domain = None
        if domain_file:
            from utils.hddl_parser import HDDLParser
            self.domain = HDDLParser.parse_domain(domain_file)

        if api_key:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.request_timeout,
                    max_retries=0,
                )
            else:
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=self.request_timeout,
                    max_retries=0,
                )

    def generate(self, nl_instruction: str):
        """
        Generate LTLf specification from natural language instruction

        Args:
            nl_instruction: Natural language instruction

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
            - prompt_dict: {"system": "...", "user": "..."}
            - response_text: raw LLM response

        Raises:
            RuntimeError: If no API key is configured
        """
        if not self.client:
            raise RuntimeError(
                "No API key configured. Please set OPENAI_API_KEY in .env file.\n"
                "Copy .env.example to .env and add your API key."
            )

        return self._generate_with_llm(nl_instruction)

    def _generate_with_llm(self, nl_instruction: str):
        """
        Generate using LLM API

        Returns:
            Tuple of (LTLSpecification, prompt_dict, response_text)
        """
        from .prompts import get_ltl_system_prompt, get_ltl_user_prompt_with_options

        stage_start = time.perf_counter()
        self.last_generation_metadata = {}
        timing_profile: dict[str, float] = {}

        # Build system prompt dynamically from the parsed domain.
        if not self.domain:
            raise RuntimeError("NLToLTLfGenerator requires parsed domain context.")

        prompt_start = time.perf_counter()
        domain_name = self.domain.name
        types_str = ', '.join(self.domain.types) if self.domain.types else 'objects'
        predicates_str = '\n'.join([f"- {pred.to_signature()}" for pred in self.domain.predicates])
        actions_str = '\n'.join([f"- {action.to_description()}" for action in self.domain.actions])
        tasks_str = '\n'.join([f"- {task.to_signature()}" for task in self.domain.tasks])

        system_prompt = get_ltl_system_prompt(
            domain_name,
            types_str,
            predicates_str,
            actions_str,
            tasks_str,
        )
        compact_task_clauses = self._extract_task_invocation_clauses(nl_instruction)
        prefer_compact_output = bool(compact_task_clauses) and (
            self._should_prefer_compact_task_grounded_output(nl_instruction)
        )
        prefer_skeletal_output = (
            prefer_compact_output
            and self._should_use_skeletal_task_grounded_output(compact_task_clauses)
        )
        user_prompt = get_ltl_user_prompt_with_options(
            nl_instruction,
            prefer_compact_task_grounded_output=prefer_compact_output,
            prefer_skeletal_task_grounded_output=prefer_skeletal_output,
            compact_task_clauses=compact_task_clauses if prefer_compact_output else (),
        )

        # Store prompt for logging
        prompt_dict = {
            "system": system_prompt,
            "user": user_prompt
        }
        timing_profile["prompt_build_seconds"] = time.perf_counter() - prompt_start

        # Call LLM API with timeout handling
        try:
            llm_request_start = time.perf_counter()
            response = self._create_chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            timing_profile["llm_roundtrip_seconds"] = (
                time.perf_counter() - llm_request_start
            )
        except Exception as e:
            timing_profile["total_seconds"] = time.perf_counter() - stage_start
            self.last_generation_metadata = {
                "timing_profile": dict(timing_profile),
                "task_clause_count": len(compact_task_clauses),
                "prefer_compact_task_grounded_output": prefer_compact_output,
                "prefer_skeletal_task_grounded_output": prefer_skeletal_output,
                "failure_stage": "llm_roundtrip",
            }
            raise RuntimeError(
                f"LLM API call failed: {type(e).__name__}: {str(e)}\n"
                f"Model: {self.model}\n"
                f"Instruction: {nl_instruction[:100]}...\n"
                f"Please check your API key, network connection, and model availability."
            ) from e

        response_extract_start = time.perf_counter()
        result_text = self._extract_response_text(response)
        timing_profile["response_extract_seconds"] = (
            time.perf_counter() - response_extract_start
        )

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            first_newline = result_text.find('\n')
            if first_newline != -1:
                closing_fence = result_text.rfind('```')
                if closing_fence != -1 and closing_fence > first_newline:
                    result_text = result_text[first_newline+1:closing_fence].strip()

        json_parse_start = time.perf_counter()
        parsed_result = self._parse_result_json(result_text)
        timing_profile["json_parse_seconds"] = time.perf_counter() - json_parse_start

        payload_normalisation_start = time.perf_counter()
        result = self._normalise_result_payload(parsed_result)
        timing_profile["payload_normalisation_seconds"] = (
            time.perf_counter() - payload_normalisation_start
        )

        # Build LTL specification
        spec_build_start = time.perf_counter()
        spec = LTLSpecification()
        spec.objects = list(result.get("objects", []))
        spec.initial_state = []
        spec.source_instruction = nl_instruction
        spec.negation_hints = {
            "policy": "all_naf",
        }

        # Helper function to recursively build formula structures
        def build_formula_recursive(formula_def):
            """Recursively build LTLFormula from nested JSON structure"""
            if not isinstance(formula_def, dict):
                return None

            # Check if this is a nested formula type
            if "type" in formula_def:
                formula_type = formula_def["type"]

                if formula_type == "negation":
                    neg_formula = build_formula_recursive(formula_def["formula"])
                    if neg_formula is None:
                        # It's an atomic predicate
                        neg_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[neg_formula],
                        logical_op=LogicalOperator.NOT
                    )

                elif formula_type == "conjunction":
                    conjuncts = []
                    for f_def in formula_def["formulas"]:
                        sub_formula = build_formula_recursive(f_def)
                        if sub_formula is None:
                            # It's an atomic predicate
                            sub_formula = LTLFormula(
                                operator=None,
                                predicate=f_def,
                                sub_formulas=[],
                                logical_op=None
                            )
                        conjuncts.append(sub_formula)
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=conjuncts,
                        logical_op=LogicalOperator.AND
                    )

                elif formula_type == "disjunction":
                    disjuncts = []
                    for f_def in formula_def["formulas"]:
                        sub_formula = build_formula_recursive(f_def)
                        if sub_formula is None:
                            # It's an atomic predicate
                            sub_formula = LTLFormula(
                                operator=None,
                                predicate=f_def,
                                sub_formulas=[],
                                logical_op=None
                            )
                        disjuncts.append(sub_formula)
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=disjuncts,
                        logical_op=LogicalOperator.OR
                    )

                elif formula_type == "implication":
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=LogicalOperator.IMPLIES
                    )

                elif formula_type == "equivalence":
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=None,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=LogicalOperator.EQUIVALENCE
                    )

                elif formula_type == "temporal":
                    # Nested temporal (e.g., F inside implication)
                    operator = TemporalOperator(formula_def["operator"])
                    inner = build_formula_recursive(formula_def["formula"])
                    if inner is None:
                        inner = LTLFormula(
                            operator=None,
                            predicate=formula_def["formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=operator,
                        predicate=None,
                        sub_formulas=[inner],
                        logical_op=None
                    )

                elif formula_type == "until":
                    # Until operator (binary temporal)
                    operator = TemporalOperator.UNTIL
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=operator,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=None
                    )

                elif formula_type == "release":
                    # Release operator (binary temporal)
                    operator = TemporalOperator.RELEASE
                    left_formula = build_formula_recursive(formula_def["left_formula"])
                    right_formula = build_formula_recursive(formula_def["right_formula"])
                    if left_formula is None:
                        left_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["left_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    if right_formula is None:
                        right_formula = LTLFormula(
                            operator=None,
                            predicate=formula_def["right_formula"],
                            sub_formulas=[],
                            logical_op=None
                        )
                    return LTLFormula(
                        operator=operator,
                        predicate=None,
                        sub_formulas=[left_formula, right_formula],
                        logical_op=None
                    )

                elif formula_type == "nested":
                    # Nested temporal operators (e.g., F(G(...)))
                    outer_op = TemporalOperator(formula_def["outer_operator"])
                    inner_op = TemporalOperator(formula_def["inner_operator"])

                    # Build innermost formula recursively
                    innermost = build_formula_recursive(formula_def["formula"])
                    if innermost is None:
                        innermost = LTLFormula(
                            operator=None,
                            predicate=formula_def["formula"],
                            sub_formulas=[],
                            logical_op=None
                        )

                    # Create inner temporal formula
                    inner_formula = LTLFormula(
                        operator=inner_op,
                        predicate=None,
                        sub_formulas=[innermost],
                        logical_op=None
                    )

                    # Create outer temporal formula
                    return LTLFormula(
                        operator=outer_op,
                        predicate=None,
                        sub_formulas=[inner_formula],
                        logical_op=None
                    )

            # Not a special type - return None to indicate it's an atomic predicate
            return None

        # Convert formulas
        for ltl_def in result["ltl_formulas"]:
            # First try to use recursive builder for any complex structure
            formula = build_formula_recursive(ltl_def)
            if formula is not None:
                spec.add_formula(formula)
                continue

            # Fallback: handle specific types
            if ltl_def["type"] == "temporal":
                operator = TemporalOperator(ltl_def["operator"])
                inner_formula_def = ltl_def["formula"]

                # Use recursive builder
                atomic = build_formula_recursive(inner_formula_def)
                if atomic is None:
                    # It's an atomic predicate
                    atomic = LTLFormula(
                        operator=None,
                        predicate=inner_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[atomic],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "until":
                operator = TemporalOperator.UNTIL
                left_formula_def = ltl_def["left_formula"]
                right_formula_def = ltl_def["right_formula"]

                # Use recursive builder
                left_formula = build_formula_recursive(left_formula_def)
                if left_formula is None:
                    left_formula = LTLFormula(
                        operator=None,
                        predicate=left_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                right_formula = build_formula_recursive(right_formula_def)
                if right_formula is None:
                    right_formula = LTLFormula(
                        operator=None,
                        predicate=right_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[left_formula, right_formula],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "release":
                operator = TemporalOperator.RELEASE
                left_formula_def = ltl_def["left_formula"]
                right_formula_def = ltl_def["right_formula"]

                # Use recursive builder
                left_formula = build_formula_recursive(left_formula_def)
                if left_formula is None:
                    left_formula = LTLFormula(
                        operator=None,
                        predicate=left_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                right_formula = build_formula_recursive(right_formula_def)
                if right_formula is None:
                    right_formula = LTLFormula(
                        operator=None,
                        predicate=right_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                formula = LTLFormula(
                    operator=operator,
                    predicate=None,
                    sub_formulas=[left_formula, right_formula],
                    logical_op=None
                )

                spec.add_formula(formula)

            elif ltl_def["type"] == "nested":
                outer_op = TemporalOperator(ltl_def["outer_operator"])
                inner_op = TemporalOperator(ltl_def["inner_operator"])
                inner_formula_def = ltl_def["formula"]

                # Use recursive builder for the innermost formula
                atomic = build_formula_recursive(inner_formula_def)
                if atomic is None:
                    atomic = LTLFormula(
                        operator=None,
                        predicate=inner_formula_def,
                        sub_formulas=[],
                        logical_op=None
                    )

                inner_formula = LTLFormula(
                    operator=inner_op,
                    predicate=None,
                    sub_formulas=[atomic],
                    logical_op=None
                )

                outer_formula = LTLFormula(
                    operator=outer_op,
                    predicate=None,
                    sub_formulas=[inner_formula],
                    logical_op=None
                )

                spec.add_formula(outer_formula)

        spec.objects = self._augment_objects_from_formulas_and_atoms(
            spec.objects,
            spec.formulas,
            result.get("atoms"),
        )

        # Create grounding map - use LLM-provided atoms if available, otherwise extract
        if "atoms" in result and result["atoms"]:
            spec.grounding_map = self._create_grounding_map_from_atoms(result["atoms"], spec)
        else:
            spec.grounding_map = self._create_grounding_map(spec)

        timing_profile["spec_build_seconds"] = time.perf_counter() - spec_build_start
        timing_profile["total_seconds"] = time.perf_counter() - stage_start
        self.last_generation_metadata = {
            "timing_profile": dict(timing_profile),
            "task_clause_count": len(compact_task_clauses),
            "prefer_compact_task_grounded_output": prefer_compact_output,
            "prefer_skeletal_task_grounded_output": prefer_skeletal_output,
        }
        return (spec, prompt_dict, result_text)

    @staticmethod
    def _normalise_result_payload(result: dict) -> dict:
        """
        Accept small schema-key drift in otherwise valid Stage 1 JSON payloads.

        Some providers keep the requested structure but rename top-level keys such
        as `ltl_formulas` -> `formulas` or `objects` -> `semantic_objects`. The
        Stage 1 contract still requires the canonical keys internally, so map a
        small set of equivalent aliases before building the specification.
        """
        if not isinstance(result, dict):
            raise RuntimeError("Stage 1 JSON payload must decode to an object.")

        normalised = dict(result)
        if "objects" not in normalised:
            for alias in ("semantic_objects", "query_objects"):
                alias_value = normalised.get(alias)
                if isinstance(alias_value, list):
                    normalised["objects"] = alias_value
                    break
        normalised.setdefault("objects", [])

        if "ltl_formulas" not in normalised:
            for alias in ("formulas", "ltlf_formulas", "temporal_formulas"):
                alias_value = normalised.get(alias)
                if isinstance(alias_value, list):
                    normalised["ltl_formulas"] = alias_value
                    break

        if "atoms" not in normalised:
            alias_value = normalised.get("grounded_atoms")
            if isinstance(alias_value, list):
                normalised["atoms"] = alias_value
        normalised.setdefault("atoms", [])
        return normalised

    def _create_chat_completion(self, messages: list[dict[str, str]]):
        """
        Request a JSON-only Stage 1 completion with a compatibility fallback.

        Prefer provider-enforced JSON output when available because Stage 1 is
        schema-constrained. Fall back to the plain-text path only when the
        backend explicitly rejects the `response_format` parameter.
        """
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "timeout": self.request_timeout,
            "max_tokens": self.response_max_tokens,
        }

        try:
            return self.client.chat.completions.create(
                response_format={"type": "json_object"},
                **request_kwargs,
            )
        except Exception as exc:
            if self._is_unsupported_json_response_format_error(exc):
                return self.client.chat.completions.create(**request_kwargs)
            raise

    def _extract_response_text(self, response: object) -> str:
        """
        Extract textual JSON content from provider-specific response shapes.

        Different chat-completion backends can return Stage 1 payloads as a plain
        string, a structured content-parts list, or a parsed JSON object. Stage 1
        needs a single textual JSON blob, so normalise those variants here and
        fail with a clear contract error if the backend returns no usable text.
        """
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
            refusal = dumped_message.get("refusal")
            refusal_text = self._normalise_response_content(refusal)
            if refusal_text:
                raise RuntimeError(f"LLM refused Stage 1 response: {refusal_text}")

        finish_reason = getattr(choices[0], "finish_reason", None)
        raise RuntimeError(
            "LLM response did not contain usable textual JSON content. "
            f"finish_reason={finish_reason!r}"
        )

    def _normalise_response_content(self, content: object) -> str | None:
        """
        Convert provider content variants into one stripped text blob.
        """
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
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content).strip() or None
        if isinstance(content, (list, tuple)):
            parts: list[str] = []
            for item in content:
                extracted = self._normalise_response_content(item)
                if extracted is not None:
                    parts.append(extracted)
            if not parts:
                return None
            return "\n".join(parts).strip() or None
        text_attr = getattr(content, "text", None)
        extracted = self._normalise_response_content(text_attr)
        if extracted is not None:
            return extracted
        value_attr = getattr(content, "value", None)
        extracted = self._normalise_response_content(value_attr)
        if extracted is not None:
            return extracted
        stringified = str(content).strip()
        return stringified or None

    @staticmethod
    def _is_unsupported_json_response_format_error(exc: Exception) -> bool:
        """
        Detect provider errors that specifically reject `response_format`.

        This keeps the fallback narrow: transport or authentication failures
        must still surface as hard errors instead of silently degrading.
        """
        message = str(exc).lower()
        if "response_format" not in message and "json_object" not in message:
            return False
        unsupported_markers = (
            "unsupported",
            "not supported",
            "invalid parameter",
            "unknown parameter",
            "unrecognized request argument",
            "extra inputs are not permitted",
        )
        return any(marker in message for marker in unsupported_markers)

    def _should_prefer_compact_task_grounded_output(self, nl_instruction: str) -> bool:
        """
        Prefer compact output for explicit declared-task queries.

        Benchmark-style instructions explicitly name declared task invocations.
        Downstream task canonicalisation already rebuilds the ordered predicate
        targets deterministically, so asking the model to emit a deeply nested
        temporal tree only increases latency and truncation risk.
        """
        if not self.domain:
            return False
        clauses = self._extract_task_invocation_clauses(nl_instruction)
        if not clauses:
            return False
        lowered = nl_instruction.lower()
        return "complete the tasks" in lowered or len(clauses) >= 4

    def _extract_task_invocation_clauses(self, nl_instruction: str) -> tuple[str, ...]:
        if not self.domain:
            return ()
        task_names = [
            getattr(task, "name", "")
            for task in getattr(self.domain, "tasks", ())
            if getattr(task, "name", "")
        ]
        if not task_names:
            return ()
        pattern = re.compile(
            r"(?P<task_name>"
            + "|".join(
                re.escape(task_name)
                for task_name in sorted(task_names, key=len, reverse=True)
            )
            + r")\((?P<args>[^()]*)\)",
        )
        clauses = []
        for match in pattern.finditer(nl_instruction):
            args_text = match.group("args").strip()
            args = [
                part.strip()
                for part in args_text.split(",")
                if part.strip()
            ]
            if args:
                clauses.append(f"{match.group('task_name')}({', '.join(args)})")
            else:
                clauses.append(f"{match.group('task_name')}()")
        return tuple(clauses)

    @staticmethod
    def _should_use_skeletal_task_grounded_output(
        compact_task_clauses: tuple[str, ...],
    ) -> bool:
        """
        Enable skeletal output when the explicit task list is too large to unroll.

        The pipeline already canonicalises explicit benchmark task lists from the
        query anchors. For very large lists, asking the model to spell out every
        shallow obligation only increases truncation risk without improving the
        canonical Stage 1 contract.
        """
        return len(tuple(compact_task_clauses or ())) >= 8

    def _parse_result_json(self, result_text: str) -> dict:
        """Parse the LLM response, tolerating prose wrappers around one JSON object."""
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as original_error:
            raw_decoded = self._decode_leading_json_object(result_text)
            if raw_decoded is not None:
                return raw_decoded
            schema_candidate = self._extract_schema_key_json_candidate(result_text)
            if schema_candidate is not None:
                try:
                    return json.loads(schema_candidate)
                except json.JSONDecodeError:
                    pass
            candidate = self._extract_json_object_candidate(result_text)
            if candidate is not None:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(
                f"Failed to parse LLM response as JSON: {str(original_error)}\n"
                f"Response preview: {result_text[:200]}...\n"
                f"The LLM did not return valid JSON. Please try again or check the prompt."
            ) from original_error

    @staticmethod
    def _extract_json_object_candidate(result_text: str) -> str | None:
        start_index = result_text.find("{")
        end_index = result_text.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return None
        candidate = result_text[start_index:end_index + 1].strip()
        return candidate or None

    @staticmethod
    def _extract_schema_key_json_candidate(result_text: str) -> str | None:
        """
        Recover a JSON object when the provider injects junk before the first schema key.

        Some Stage 1 responses begin with stray tokens such as `{"|", ...}` before
        resuming the required `objects` / `ltl_formulas` / `atoms` schema. When a
        required top-level key is visible, rebuild the candidate object from that
        key onward instead of failing immediately.
        """
        required_keys = ('"objects"', '"ltl_formulas"', '"atoms"')
        key_positions = [
            result_text.find(key)
            for key in required_keys
            if result_text.find(key) != -1
        ]
        if not key_positions:
            return None
        key_index = min(key_positions)
        end_index = result_text.rfind("}")
        if end_index == -1 or end_index <= key_index:
            return None
        candidate = "{" + result_text[key_index:end_index + 1]
        return candidate.strip() or None

    @staticmethod
    def _decode_leading_json_object(result_text: str) -> dict | None:
        """
        Decode the first complete JSON object and ignore trailing junk.

        Some providers return a valid object followed by duplicated suffix text or
        other non-JSON debris. When the first decoded value is a dictionary, that
        already satisfies Stage 1's contract.
        """
        stripped = result_text.lstrip()
        if not stripped.startswith("{"):
            return None
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _augment_objects_from_formulas_and_atoms(
        self,
        objects: list,
        formulas: list[LTLFormula],
        atoms_list: list | None,
    ) -> list:
        """
        Ensure `spec.objects` includes every constant used in formulas/atoms.

        Stage 1 LLM outputs can miss object declarations while still referencing
        those constants inside formulas. Downstream typing relies on a complete
        object universe, so we merge all observed constants deterministically.
        """
        merged = list(objects or [])
        seen = {item for item in merged if isinstance(item, str)}

        def add_object(value: object) -> None:
            if not isinstance(value, str):
                return
            token = value.strip()
            if not token or token.startswith("?"):
                return
            if token in seen:
                return
            seen.add(token)
            merged.append(token)

        for atom_dict in atoms_list or []:
            for arg in atom_dict.get("args", []) or []:
                add_object(arg)

        formula_stack = list(reversed(list(formulas or [])))
        while formula_stack:
            formula = formula_stack.pop()
            if formula.predicate and isinstance(formula.predicate, dict):
                special_keys = {"type", "formulas", "left_formula", "right_formula", "formula", "operator"}
                if all(key not in special_keys for key in formula.predicate.keys()):
                    for args in formula.predicate.values():
                        if isinstance(args, list):
                            for arg in args:
                                add_object(arg)
            sub_formulas = list(getattr(formula, "sub_formulas", ()) or ())
            formula_stack.extend(reversed(sub_formulas))

        return merged

    def _create_grounding_map(self, spec: LTLSpecification) -> GroundingMap:
        """
        Create grounding map from LTL specification

        Extracts all predicates from formulas and creates propositional symbols
        with their mappings back to original predicates and arguments.
        """
        gmap = GroundingMap()

        # Iteratively walk formula nodes so large cumulative conjunctions do not
        # depend on Python recursion depth.
        formula_stack = list(reversed(list(spec.formulas or [])))
        while formula_stack:
            formula = formula_stack.pop()
            if formula.predicate and isinstance(formula.predicate, dict):
                # Check if this is a real atomic predicate or a nested formula structure
                # Real predicates have predicate_name: [args] format
                # Nested structures have "type", "formulas", "left_formula", etc.
                special_keys = {"type", "formulas", "left_formula", "right_formula", "formula", "operator"}

                is_atomic_predicate = True
                for key in formula.predicate.keys():
                    if key in special_keys:
                        is_atomic_predicate = False
                        break

                if is_atomic_predicate:
                    # This is an atomic predicate like {"on": ["a", "b"]}
                    for pred_name, args in formula.predicate.items():
                        if isinstance(args, list):  # Ensure args is a list
                            # Create propositional symbol using grounding map's normalizer
                            symbol = create_propositional_symbol(pred_name, args, gmap.normalizer)
                            # Add to grounding map
                            gmap.add_atom(symbol, pred_name, args)

            sub_formulas = list(getattr(formula, "sub_formulas", ()) or ())
            formula_stack.extend(reversed(sub_formulas))

        return gmap

    def _create_grounding_map_from_atoms(self, atoms_list: list, spec: LTLSpecification) -> GroundingMap:
        """
        Create grounding map from LLM-provided atoms list

        Also validates that all atoms are consistent with formulas.

        Args:
            atoms_list: List of atom dicts from LLM response
            spec: LTL specification for validation

        Returns:
            GroundingMap with validated atoms
        """
        gmap = GroundingMap()

        # Add each atom from LLM response
        for atom_dict in atoms_list:
            symbol = atom_dict.get("symbol", "")
            predicate = atom_dict.get("predicate", "")
            args = atom_dict.get("args", [])

            # Validate symbol naming convention using grounding map's normalizer
            expected_symbol = create_propositional_symbol(predicate, args, gmap.normalizer)
            if symbol != expected_symbol:
                print(f"⚠️  Warning: LLM provided symbol '{symbol}' doesn't match convention '{expected_symbol}'")
                # Use the expected symbol for consistency
                symbol = expected_symbol

            gmap.add_atom(symbol, predicate, args)

        # Also extract predicates from formulas to verify completeness
        extracted_gmap = self._create_grounding_map(spec)

        # Check if LLM missed any atoms
        for extracted_symbol, extracted_atom in extracted_gmap.atoms.items():
            if extracted_symbol not in gmap.atoms:
                print(f"⚠️  Warning: LLM missed atom '{extracted_symbol}', adding it")
                gmap.add_atom(
                    extracted_symbol,
                    extracted_atom.predicate,
                    extracted_atom.args
                )

        # Validate with domain if available
        if self.domain:
            domain_predicates = [pred.name for pred in self.domain.predicates]
            for symbol, atom in gmap.atoms.items():
                errors = gmap.validate_atom(symbol, domain_predicates, spec.objects)
                if errors:
                    print(f"⚠️  Validation errors for {symbol}: {errors}")

        return gmap


def test_ltlf_generator():
    """Test LTLf generator"""
    generator = NLToLTLfGenerator()

    instruction = "Put block A on block B"
    spec, _, _ = generator.generate(instruction)

    print("="*80)
    print("LTLf Generator Test")
    print("="*80)
    print(f"Instruction: {instruction}")
    print(f"\nObjects: {spec.objects}")
    print(f"Initial State: {spec.initial_state}")
    print(f"\nLTLf Formulas:")
    for i, formula in enumerate(spec.formulas, 1):
        print(f"  {i}. {formula.to_string()}")

    print(f"\nFull Specification:")
    print(json.dumps(spec.to_dict(), indent=2))


if __name__ == "__main__":
    test_ltlf_generator()
