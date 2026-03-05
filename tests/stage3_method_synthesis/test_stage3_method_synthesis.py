"""
Focused tests for Stage 3 HTN method synthesis.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage1_interpretation.ltlf_formula import (
	LTLFormula,
	LTLSpecification,
	TemporalOperator,
)
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_prompts import (
	build_htn_system_prompt,
	build_htn_user_prompt,
)
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTask,
	HTNTargetTaskBinding,
)
from utils.config import get_config
from utils.hddl_parser import HDDLParser


def _domain():
	domain_path = (
		Path(__file__).parent.parent.parent
		/ "src"
		/ "domains"
		/ "blocksworld"
		/ "domain.hddl"
	)
	return HDDLParser.parse_domain(str(domain_path))


def _live_stage3_kwargs() -> dict:
	config = get_config()
	if not config.validate():
		pytest.skip("Stage 3 live tests require a valid OPENAI_API_KEY")

	return {
		"api_key": config.openai_api_key,
		"model": config.openai_model,
		"base_url": config.openai_base_url,
		"timeout": float(config.openai_timeout),
	}


def _atomic_formula(predicate: str, args: list[str]) -> LTLFormula:
	return LTLFormula(
		operator=None,
		predicate={predicate: args},
		sub_formulas=[],
		logical_op=None,
	)


def _eventually_on_spec() -> LTLSpecification:
	spec = LTLSpecification()
	spec.objects = ["a", "b"]
	spec.grounding_map = GroundingMap()
	spec.grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	spec.formulas = [
		LTLFormula(
			operator=TemporalOperator.FINALLY,
			predicate=None,
			sub_formulas=[_atomic_formula("on", ["a", "b"])],
			logical_op=None,
		),
	]
	return spec


def _dfa_result_for_labels(*labels: str) -> dict:
	edges = "\n".join(f'  0 -> 1 [label="{label}"];' for label in labels)
	return {
		"dfa_dot": (
			"digraph {\n"
			"  0 [shape=circle];\n"
			"  1 [shape=doublecircle];\n"
			f"{edges}\n"
			"}\n"
		),
	}


def test_extract_target_literals_discards_non_progressing_transitions():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("clear_a", "clear", ["a"])
	grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	grounding_map.add_atom("on_b_c", "on", ["b", "c"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 3;\n"
			"  node [shape = circle]; 1 2 s1 s2;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> 2 [label=\"!clear_a\"];\n"
			"  1 -> s1 [label=\"clear_a\"];\n"
			"  2 -> 2 [label=\"!clear_a\"];\n"
			"  2 -> 2 [label=\"clear_a\"];\n"
			"  3 -> 3 [label=\"!clear_a\"];\n"
			"  3 -> 3 [label=\"clear_a\"];\n"
			"  s1 -> 2 [label=\"!on_a_b\"];\n"
			"  s1 -> s2 [label=\"on_a_b\"];\n"
			"  s2 -> 2 [label=\"!on_b_c\"];\n"
			"  s2 -> 3 [label=\"on_b_c\"];\n"
			"}\n"
		),
	}

	literals = synthesizer.extract_target_literals(grounding_map, dfa_result)

	assert [literal.to_signature() for literal in literals] == [
		"clear(a)",
		"on(a, b)",
		"on(b, c)",
	]

	transition_specs = synthesizer.extract_progressing_transitions(grounding_map, dfa_result)

	assert [
		(
			spec["transition_name"],
			spec["source_state"],
			spec["target_state"],
			spec["label"],
		)
		for spec in transition_specs
	] == [
		("dfa_step_q1_q2_clear_a", "q1", "q2", "clear(a)"),
		("dfa_step_q2_q3_on_a_b", "q2", "q3", "on(a, b)"),
		("dfa_step_q3_q4_on_b_c", "q3", "q4", "on(b, c)"),
	]


def test_extract_target_literals_keeps_accepting_loops_when_no_progress_edge_exists():
	synthesizer = HTNMethodSynthesizer()
	grounding_map = GroundingMap()
	grounding_map.add_atom("clear_a", "clear", ["a"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  0 [shape = doublecircle];\n"
			"  1 [shape = circle];\n"
			"  0 -> 0 [label=\"clear_a\"];\n"
			"  0 -> 1 [label=\"!clear_a\"];\n"
			"  1 -> 1 [label=\"clear_a\"];\n"
			"  1 -> 1 [label=\"!clear_a\"];\n"
			"}\n"
		),
	}

	literals = synthesizer.extract_target_literals(grounding_map, dfa_result)

	assert [literal.to_signature() for literal in literals] == ["clear(a)"]


def test_method_synthesizer_uses_live_llm_output():
	domain = _domain()
	spec = _eventually_on_spec()
	dfa_result = _dfa_result_for_labels("on_a_b")

	library, metadata = HTNMethodSynthesizer(
		**_live_stage3_kwargs(),
	).synthesize(
		domain=domain,
		grounding_map=spec.grounding_map,
		dfa_result=dfa_result,
	)

	assert metadata["used_llm"] is True
	assert metadata["llm_prompt"] is not None
	assert metadata["llm_response"]
	assert metadata["target_literals"] == ["on(a, b)"]
	assert metadata["compound_tasks"] >= 1
	assert metadata["methods"] >= 1
	assert library.compound_tasks
	assert library.methods
	assert library.target_task_bindings

	primitive_task_names = {task.name for task in library.primitive_tasks}
	assert primitive_task_names == {
		"pick_up",
		"pick_up_from_table",
		"put_on_block",
		"put_down",
	}

	compound_task_names = {task.name for task in library.compound_tasks}
	assert all(method.task_name in compound_task_names for method in library.methods)
	assert all(
		not task_name.startswith(("achieve_", "maintain_not_"))
		for task_name in compound_task_names
	)

	bound_task = library.task_name_for_literal(library.target_literals[0])
	assert bound_task in compound_task_names


def test_method_synthesizer_requires_target_task_binding_for_each_target_literal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
	)

	with pytest.raises(ValueError, match="missing a target_task_binding"):
		synthesizer._validate_library(library, domain)


def test_target_guard_completion_adds_missing_already_satisfied_method():
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		methods=[
			HTNMethod(
				method_name="m_place_on_direct",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
					HTNLiteral("clear", ("BLOCK2",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
					),
				),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	completed_library, generated_count = synthesizer._complete_missing_target_guard_methods(library)

	assert generated_count == 1
	generated_methods = [
		method
		for method in completed_library.methods
		if method.origin == "stage3_target_guard_completion"
	]
	assert len(generated_methods) == 1
	assert generated_methods[0].task_name == "place_on"
	assert generated_methods[0].context == (
		HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
	)
	assert generated_methods[0].subtasks == ()


def test_stage3_prompts_make_binding_and_naming_rules_explicit():
	system_prompt = build_htn_system_prompt()
	user_prompt = build_htn_user_prompt(
		_domain(),
		["on(a, b)", "!clear(a)"],
		'{"target_task_bindings": [], "compound_tasks": [], "methods": []}',
	)

	assert "Always include exactly these top-level keys: target_task_bindings, compound_tasks, methods." in system_prompt
	assert "method_name must follow exactly: m_{task_name}_{strategy}." in system_prompt
	assert "Never use prefixes achieve_, maintain_not_, ensure_, or goal_." in system_prompt
	assert "ordering must be a list of edges [from_step_id, to_step_id]." in system_prompt
	assert "Do an internal two-pass review before returning" in system_prompt
	assert "Internally reason by explicit case partitioning" in system_prompt
	assert "Do not overfit to one witness plan or one canonical initial state." in system_prompt
	assert "Do not rely on later stages to invent missing helper-task branches." in system_prompt
	assert "Never introduce free variables in subtasks." in system_prompt
	assert "Respect type discipline." in system_prompt
	assert "If two roles must stay distinct" in system_prompt
	assert "Use stable upper-case variable names derived from the domain types" in system_prompt
	assert "If a method needs an extra local helper variable beyond the task arguments" in system_prompt
	assert "Constructive sibling methods for the same task must be semantically distinguishable" in system_prompt
	assert "A negative literal must bind to a task whose methods remove, prevent, or keep the predicate false." in system_prompt
	assert "Do not invent an extra sibling method whose only difference is that it performs extra prerequisite helper steps" in system_prompt
	assert "Never reveal chain-of-thought, scratch work, analysis, or self-check text." in system_prompt
	assert "Do not emit duplicate target_task_bindings entries, duplicate compound task " in system_prompt
	assert "Sibling methods for the same task may exist and are expected" in system_prompt
	assert "same sibling branch body under different method names for the same task" in system_prompt

	assert "DOMAIN TYPES:" in user_prompt
	assert "REQUIRED target_task_bindings ENTRIES:" in user_prompt
	assert '{"target_literal": "on(a, b)", "task_name": "<semantic_task_name>"}' in user_prompt
	assert "Think in explicit case splits before returning." in user_prompt
	assert "Do not use achieve_, maintain_not_, ensure_, or goal_ prefixes anywhere in compound task names." in user_prompt
	assert "For every method, method_name must be exactly 'm_' + task_name + '_' + a short strategy suffix." in user_prompt
	assert "For a single-step method, ordering may be []." in user_prompt
	assert "Do all methods use the exact m_task_strategy naming pattern?" in user_prompt
	assert "Do not overfit methods to the default all-objects-on-table example state." in user_prompt
	assert "Use the declared HDDL types and typed action signatures exactly." in user_prompt
	assert "Do not collapse two distinct semantic roles onto one variable" in user_prompt
	assert "introduce and bind the actual blocker/support object as a separate variable" in user_prompt
	assert "Use is_positive=true for equality and is_positive=false for disequality." in user_prompt
	assert "Do not assume PANDA, the renderer, or any later stage will synthesize missing branches for you." in user_prompt
	assert "For every helper task that denotes a reusable stateful intention" in user_prompt
	assert "Think twice before returning: first verify the JSON shape, then verify task coverage." in user_prompt
	assert "Never reveal chain-of-thought, hidden reasoning, self-critique, or analysis text." in user_prompt
	assert "Do not duplicate target_task_bindings entries or compound task declarations." in user_prompt
	assert "Multiple methods for the same task are normal when they express different sibling " in user_prompt
	assert "Do not invent free variables such as TOP, SUPPORT, X, or Y" in user_prompt
	assert "If you introduce an extra local variable in method.parameters" in user_prompt
	assert "Do not rename an existing task parameter to a fresh variable." in user_prompt
	assert "Prefer type-based upper-case variable names in the final JSON." in user_prompt
	assert "Those sibling strategy methods must be distinguishable by reusable method-level context." in user_prompt
	assert "Never bind a negative literal to a task whose constructive branch makes the positive relation true." in user_prompt
	assert "INVALID target_task_bindings entry:" in user_prompt
	assert "Why invalid: link_nodes constructively makes linked(...) true" in user_prompt
	assert "Do not create a generic clear_first or prepare_first sibling" in user_prompt
	assert "Invalid pattern: one sibling says acquire, another says stack" in user_prompt
	assert "Did you avoid every unbound free variable in subtasks" in user_prompt
	assert "Did you explicitly cover the already-satisfied, direct, blocked, and recursive-helper case families" in user_prompt
	assert "Did every variable and subtask argument respect the domain's declared types?" in user_prompt
	assert "Did every equality/disequality constraint use the supported literal form" in user_prompt
	assert "Did you keep existing task parameters stable instead of renaming them" in user_prompt
	assert "Did every constructive sibling method for the same task expose a distinguishable" in user_prompt
	assert "Did you use stable upper-case type-based variable names" in user_prompt
	assert "Did every target_task_binding use a task whose semantics match the target literal's polarity" in user_prompt
	assert "Did you avoid adding redundant clear_first/prepare_first siblings" in user_prompt
	assert "Did you avoid printing any chain-of-thought, analysis, or self-check prose" in user_prompt
	assert "Did you avoid duplicate binding entries, duplicate compound task declarations, and " in user_prompt


def test_method_synthesizer_rejects_llm_identifiers_that_need_silent_sanitising():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place-on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place-on_noop",
				task_name="place-on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	assert normalised.compound_tasks[0].name == "place-on"

	with pytest.raises(ValueError, match="Invalid task identifier 'place-on'"):
		synthesizer._validate_library(normalised, domain)


def test_normalise_llm_library_rewrites_primitive_action_name_to_source_hddl_name():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
						action_name=None,
						literal=None,
						preconditions=(),
						effects=(),
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("B",),
						kind="primitive",
						action_name="pick_up_from_table",
						literal=None,
						preconditions=(),
						effects=(),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)
	primitive_step = normalised.methods[0].subtasks[1]

	assert primitive_step.task_name == "pick_up_from_table"
	assert primitive_step.action_name == "pick-up-from-table"


def test_normalise_llm_library_promotes_used_local_variables_into_method_parameters():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("make_not_clear", ("BLOCK",), False, ("clear",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_make_not_clear_acquire",
				task_name="make_not_clear",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK"),
						kind="primitive",
						action_name="put_on_block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	normalised = synthesizer._normalise_llm_library(library, domain)

	assert normalised.methods[0].parameters == ("BLOCK", "BLOCK1")


def test_method_validation_rejects_legacy_task_prefixes():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("achieve_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_achieve_on_stack",
				task_name="achieve_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="Legacy task name 'achieve_on' is not allowed"):
		synthesizer._validate_library(library, domain)


def test_method_validation_enforces_method_name_contract():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="place_on_stack",
				task_name="place_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="must follow the exact naming pattern"):
		synthesizer._validate_library(library, domain)


def test_negative_target_requires_constructive_method():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_not_clear", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_not_clear_noop",
				task_name="keep_not_clear",
				parameters=("B",),
				context=(
					HTNLiteral("clear", ("B",), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("clear", ("a",), False, "clear_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!clear(a)", "keep_not_clear"),
		],
	)

	with pytest.raises(ValueError, match="has no constructive non-zero-subtask method"):
		synthesizer._validate_library(library, domain)


def test_negative_target_binding_must_match_negative_semantics():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
						literal=HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), False, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "place_on"),
		],
	)

	with pytest.raises(ValueError, match="none of that task's methods exposes an already-satisfied context"):
		synthesizer._validate_library(library, domain)


def test_synthesize_injects_strong_negation_mode_and_tilde_signature(monkeypatch):
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	synthesizer.client = object()

	grounding_map = GroundingMap()
	grounding_map.add_atom("on_a_b", "on", ["a", "b"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 1;\n"
			"  node [shape = circle]; 0;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 0;\n"
			"  0 -> 1 [label=\"!on_a_b\"];\n"
			"}\n"
		),
	}

	llm_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("B1", "B2"), False, ("on",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(
					HTNLiteral("on", ("B1", "B2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_pickup",
				task_name="remove_on",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1", "B2"),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[
			HTNTargetTaskBinding("!on(a, b)", "remove_on"),
		],
	)

	def _fake_request_complete_llm_library(prompt, domain_obj, metadata):
		return llm_library, '{"ok": true}', "stop"

	monkeypatch.setattr(
		synthesizer,
		"_request_complete_llm_library",
		_fake_request_complete_llm_library,
	)

	library, metadata = synthesizer.synthesize(
		domain=domain,
		grounding_map=grounding_map,
		dfa_result=dfa_result,
		query_text="Keep on(a,b) explicitly false.",
	)

	assert library.target_literals[0].negation_mode == "strong"
	assert library.target_literals[0].to_signature() == "~on(a, b)"
	assert metadata["target_literals"] == ["~on(a, b)"]
	assert metadata["negation_resolution"]["mode_by_predicate"] == {"on/2": "strong"}
	assert library.target_task_bindings[0].target_literal == "~on(a, b)"


def test_method_validation_rejects_unbound_free_variables_in_subtasks():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="uses unbound variable 'TOP'"):
		synthesizer._validate_library(library, domain)


def test_method_validation_allows_local_variables_when_bound_in_context():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("clear_top", ("B",), False, ("clear",)),
			HTNTask("hold_block", ("B",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_clear_top_recursive",
				task_name="clear_top",
				parameters=("B",),
				context=(
					HTNLiteral("on", ("TOP", "B"), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("TOP",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_down",
						args=("TOP",),
						kind="primitive",
						action_name="put-down",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_method_validation_allows_local_variables_when_bound_in_preconditions():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B",), False, ("holding",)),
			HTNTask("clear_top", ("B",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("B",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up",
						args=("B", "SUPPORT"),
						kind="primitive",
						action_name="pick-up",
						preconditions=(
							HTNLiteral("on", ("B", "SUPPORT"), True, None),
						),
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_primitive_alias_cannot_use_non_primitive_subtask_kind():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("B1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("B1",),
				context=(
					HTNLiteral("holding", ("B1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("B1", "B2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("B1", "B2"),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="Primitive aliases must use kind='primitive'"):
		synthesizer._validate_library(library, domain)


def test_sibling_constructive_methods_must_have_distinguishable_contexts():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("holding", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_table_again",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("holding", ("a",), True, "holding_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("holding(a)", "hold_block"),
		],
	)

	with pytest.raises(ValueError, match="semantically duplicate|not semantically distinguishable"):
		synthesizer._validate_library(library, domain)


def test_redundant_constructive_siblings_are_pruned_before_validation():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK1",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_noop",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_acquire",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
						action_name="put-on-block",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_noop",
				task_name="hold_block",
				parameters=("BLOCK1",),
				context=(
					HTNLiteral("holding", ("BLOCK1",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("on", ("a", "b"), True, "on_a_b"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("on(a, b)", "place_on"),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 1
	assert {method.method_name for method in pruned_library.methods} == {
		"m_place_on_noop",
		"m_place_on_stack",
		"m_hold_block_noop",
	}
	synthesizer._validate_library(pruned_library, domain)


def test_direct_self_recursive_siblings_are_pruned_when_non_recursive_alternative_exists():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
			HTNTask("clear_top", ("BLOCK",), False, ("clear",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_hold_block_from_table",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("ontable", ("BLOCK",), True, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up_from_table",
						args=("BLOCK",),
						kind="primitive",
						action_name="pick-up-from-table",
					),
				),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_clear_first",
				task_name="hold_block",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="clear_top",
						args=("BLOCK",),
						kind="compound",
					),
					HTNMethodStep(
						step_id="s2",
						task_name="hold_block",
						args=("BLOCK",),
						kind="compound",
					),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_clear_top_noop",
				task_name="clear_top",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("clear", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
	)

	pruned_library, pruned_count = synthesizer._prune_redundant_constructive_siblings(
		library,
		domain,
	)

	assert pruned_count == 1
	assert {method.method_name for method in pruned_library.methods} == {
		"m_hold_block_from_table",
		"m_clear_top_noop",
	}


def test_method_validation_rejects_conflicting_variable_types():
	domain = SimpleNamespace(
		name="typed_domain",
		actions=[
			SimpleNamespace(
				name="place",
				parameters=["?item - block", "?slot - location"],
				preconditions="(and)",
				effects="(and (stored ?item))",
			),
		],
		predicates=[
			SimpleNamespace(
				name="stored",
				parameters=["?item - block"],
				to_signature=lambda: "stored(?item - block)",
			),
		],
		requirements=[],
		types=["block", "location"],
	)
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("misbind", ("BLOCK",), False, ("stored",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_misbind_noop",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(
					HTNLiteral("stored", ("BLOCK",), True, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_misbind_conflict",
				task_name="misbind",
				parameters=("BLOCK",),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="place",
						args=("BLOCK", "BLOCK"),
						kind="primitive",
						action_name="place",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[
			HTNLiteral("stored", ("a",), True, "stored_a"),
		],
		target_task_bindings=[
			HTNTargetTaskBinding("stored(a)", "misbind"),
		],
	)

	with pytest.raises(ValueError, match="conflicting inferred types"):
		synthesizer._validate_library(library, domain)


def test_method_validation_accepts_supported_equality_constraints():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("keep_apart", ("BLOCK1", "BLOCK2"), False, ()),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_keep_apart_distinct",
				task_name="keep_apart",
				parameters=("BLOCK1", "BLOCK2"),
				context=(
					HTNLiteral("=", ("BLOCK1", "BLOCK2"), False, None),
				),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	synthesizer._validate_library(library, domain)


def test_parse_llm_library_rejects_truncated_json_with_clear_error():
	synthesizer = HTNMethodSynthesizer()

	with pytest.raises(ValueError, match="appears truncated"):
		synthesizer._parse_llm_library(
			'{"target_task_bindings": [], "compound_tasks": [',
		)


def test_validate_library_rejects_semantically_duplicate_methods():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("place_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_place_on_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_place_on_duplicate_stack",
				task_name="place_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("holding", ("BLOCK1",), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="put_on_block",
						args=("BLOCK1", "BLOCK2"),
						kind="primitive",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	with pytest.raises(ValueError, match="semantically duplicate"):
		synthesizer._validate_library(library, domain)


def test_request_complete_llm_library_retries_until_full_json():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer(max_attempts=2)
	prompt = {"system": "system", "user": "user"}
	metadata = {"llm_attempts": 0}
	valid_payload = json.dumps(
		{
			"compound_tasks": [
				{
					"name": "place_on",
					"parameters": ["BLOCK1", "BLOCK2"],
					"is_primitive": False,
					"source_predicates": ["on"],
				},
			],
			"methods": [
				{
					"method_name": "m_place_on_noop",
					"task_name": "place_on",
					"parameters": ["BLOCK1", "BLOCK2"],
					"context": [
						{
							"predicate": "on",
							"args": ["BLOCK1", "BLOCK2"],
							"is_positive": True,
						},
					],
					"subtasks": [],
					"ordering": [],
					"origin": "llm",
				},
			],
			"target_task_bindings": [
				{"target_literal": "on(a, b)", "task_name": "place_on"},
			],
		},
	)
	responses = iter(
		[
			('{"compound_tasks": [', "length"),
			(valid_payload, "stop"),
		],
	)

	def fake_call_llm(
		prompt_payload: dict,
		*,
		retry_instruction: str | None = None,
		max_tokens: int | None = None,
	):
		return next(responses)

	synthesizer._call_llm = fake_call_llm  # type: ignore[method-assign]

	library, response_text, finish_reason = synthesizer._request_complete_llm_library(
		prompt,
		domain,
		metadata,
	)

	assert library.compound_tasks
	assert response_text == valid_payload
	assert finish_reason == "stop"
	assert metadata["llm_attempts"] == 2
	assert len(metadata["llm_attempt_durations_seconds"]) == 2
	assert metadata["llm_response_time_seconds"] >= 0


def test_negative_target_binding_accepts_helper_mediated_removal():
	domain = _domain()
	synthesizer = HTNMethodSynthesizer()
	library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("remove_on", ("BLOCK1", "BLOCK2"), False, ("on",)),
			HTNTask("hold_block", ("BLOCK",), False, ("holding",)),
		],
		primitive_tasks=synthesizer._build_primitive_tasks(domain),
		methods=[
			HTNMethod(
				method_name="m_remove_on_noop",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), False, None),),
				subtasks=(),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_remove_on_from_block",
				task_name="remove_on",
				parameters=("BLOCK1", "BLOCK2"),
				context=(HTNLiteral("on", ("BLOCK1", "BLOCK2"), True, None),),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="hold_block",
						args=("BLOCK1",),
						kind="compound",
					),
				),
				ordering=(),
				origin="llm",
			),
			HTNMethod(
				method_name="m_hold_block_from_block",
				task_name="hold_block",
				parameters=("BLOCK", "SUPPORT"),
				context=(
					HTNLiteral("on", ("BLOCK", "SUPPORT"), True, None),
					HTNLiteral("clear", ("BLOCK",), True, None),
					HTNLiteral("handempty", (), True, None),
					HTNLiteral("=", ("BLOCK", "SUPPORT"), False, None),
				),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="pick_up",
						args=("BLOCK", "SUPPORT"),
						kind="primitive",
						action_name="pick-up",
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), False, "on_a_b")],
		target_task_bindings=[HTNTargetTaskBinding("!on(a, b)", "remove_on")],
	)

	synthesizer._validate_library(library, domain)
