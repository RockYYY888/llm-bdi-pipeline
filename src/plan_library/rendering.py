"""
Render structured AgentSpeak(L) plan libraries as textual `.asl` programs.
"""

from __future__ import annotations

import re
from typing import List

from method_library.synthesis.naming import sanitize_identifier

from .models import AgentSpeakBodyStep, AgentSpeakPlan, PlanLibrary


def render_plan_library_asl(plan_library: PlanLibrary) -> str:
	"""Render a structured plan library into a readable AgentSpeak(L) file."""

	lines: List[str] = [
		"/* Generated AgentSpeak(L) Plan Library */",
		f"/* Domain: {plan_library.domain_name} */",
		"",
	]
	for plan in plan_library.plans:
		lines.extend(_render_plan(plan))
		lines.append("")
	return "\n".join(lines).rstrip() + "\n"


def _render_plan(plan: AgentSpeakPlan) -> List[str]:
	trigger = _call(plan.trigger.symbol, tuple(_raw_argument(argument) for argument in plan.trigger.arguments))
	context = " & ".join(_render_context_literal(literal) for literal in plan.context) or "true"
	body_items = [_render_body_step(step) for step in plan.body]
	if not body_items:
		body_items = ["true"]
	source_ids = ", ".join(plan.source_instruction_ids) if plan.source_instruction_ids else "none"
	lines = [
		f"/* plan={plan.plan_name} | source_instruction_ids={source_ids} */",
		f"+!{trigger} : {context} <-",
	]
	for index, body_item in enumerate(body_items):
		suffix = ";" if index < len(body_items) - 1 else "."
		lines.append(f"\t{body_item}{suffix}")
	return lines


def _render_body_step(step: AgentSpeakBodyStep) -> str:
	call = _call(step.symbol, step.arguments)
	if step.kind == "subgoal":
		return f"!{call}"
	return call


def _call(symbol: str, arguments) -> str:
	if not arguments:
		return _jason_functor(symbol)
	return f"{_jason_functor(symbol)}({', '.join(_render_term(argument) for argument in arguments)})"


def _raw_argument(argument: str) -> str:
	text = str(argument or "").strip()
	if ":" in text:
		return text.split(":", 1)[0].strip()
	return text


def _render_context_literal(literal: str) -> str:
	text = str(literal or "").strip()
	if not text:
		return "true"
	if text.startswith("!"):
		return f"not {_render_atom(text[1:].strip())}"
	if text.lower().startswith("not "):
		return f"not {_render_atom(text[4:].strip())}"
	if "!=" in text:
		left, right = text.split("!=", 1)
		return f"{_render_term(left)} \\\\== {_render_term(right)}"
	if "==" in text:
		left, right = text.split("==", 1)
		return f"{_render_term(left)} == {_render_term(right)}"
	return _render_atom(text)


def _render_atom(atom: str) -> str:
	text = str(atom or "").strip()
	if not text:
		return "true"
	if "(" not in text:
		return _jason_functor(text)
	if not text.endswith(")"):
		return _jason_functor(text)
	symbol, raw_args = text.split("(", 1)
	args = tuple(argument.strip() for argument in raw_args[:-1].split(",") if argument.strip())
	return _call(symbol, args)


def _render_term(term: str) -> str:
	text = str(term or "").strip()
	if not text:
		return "item"
	if _is_agentspeak_variable(text):
		return text
	if re.fullmatch(r"[a-z][A-Za-z0-9_]*", text):
		return text
	return sanitize_identifier(text)


def _jason_functor(symbol: str) -> str:
	return sanitize_identifier(str(symbol or "").strip())


def _is_agentspeak_variable(token: str) -> bool:
	return re.fullmatch(r"[A-Z][A-Za-z0-9_]*", str(token or "").strip()) is not None
