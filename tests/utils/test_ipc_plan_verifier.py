from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage1_interpretation.grounding_map import GroundingMap
from stage3_method_synthesis.htn_method_synthesis import HTNMethodSynthesizer
from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
)
from utils.hddl_parser import HDDLParser
from utils.ipc_plan_verifier import IPCPlanVerifier


def test_render_primitive_only_plan_uses_official_primitive_plan_format():
	plan_text = IPCPlanVerifier.render_primitive_only_plan(
		[
			"unstack(b2,b3)",
			"put-down(b2)",
			"stack(b1,b4)",
		],
	)

	assert plan_text == "\n".join(
		[
			"==>",
			"0 unstack b2 b3",
			"1 put-down b2",
			"2 stack b1 b4",
			"root",
		],
	) + "\n"


def test_parse_verifier_summary_distinguishes_primitive_success_from_full_htn_success():
	output = "\n".join(
		[
			"\u001b[0;34mPrimitive plan only (only valid if there are no method effects) ...\u001b[0m",
			"Primitive plan alone executable: true",
			"Plan verification result: false",
		],
	)

	clean = IPCPlanVerifier.strip_ansi(output)

	assert IPCPlanVerifier._extract_bool(clean, "Primitive plan alone executable") is True
	assert IPCPlanVerifier._extract_bool(clean, "Plan verification result") is False
	assert IPCPlanVerifier._infer_goal_reached(clean) is True


def test_parse_verifier_summary_accepts_hierarchical_executability_marker():
	output = "\n".join(
		[
			"Plan is executable: true",
			"Plan verification result: true",
		],
	)

	assert IPCPlanVerifier._extract_executability(output) is True
	assert IPCPlanVerifier._infer_goal_reached(output) is True


def test_render_supported_hierarchical_plan_contains_root_and_decompositions():
	root = Path(__file__).resolve().parents[2]
	verifier = IPCPlanVerifier()
	domain_file = root / "src/domains/blocksworld/domain.hddl"
	domain = HDDLParser.parse_domain(str(domain_file))
	grounding_map = GroundingMap()
	grounding_map.add_atom("on_b4_b2", "on", ["b4", "b2"])
	grounding_map.add_atom("on_b1_b4", "on", ["b1", "b4"])
	grounding_map.add_atom("on_b3_b1", "on", ["b3", "b1"])
	dfa_result = {
		"dfa_dot": (
			"digraph MONA_DFA {\n"
			"  node [shape = doublecircle]; 3;\n"
			"  node [shape = circle]; 1 s1 s2;\n"
			"  init [shape = plaintext, label = \"\"];\n"
			"  init -> 1;\n"
			"  1 -> s1 [label=\"on_b1_b4\"];\n"
			"  s1 -> s2 [label=\"on_b3_b1\"];\n"
			"  s2 -> 3 [label=\"on_b4_b2\"];\n"
			"}\n"
		),
	}
	method_library, _ = HTNMethodSynthesizer().synthesize(
		domain=domain,
		grounding_map=grounding_map,
		dfa_result=dfa_result,
		ordered_literal_signatures=["on(b4, b2)", "on(b1, b4)", "on(b3, b1)"],
	)

	plan_text = verifier._render_supported_hierarchical_plan(
		domain_file=domain_file,
		problem_file=root / "src/domains/blocksworld/problems/p01.hddl",
		action_path=[
			"nop()",
			"unstack(b2,b3)",
			"put-down(b2)",
			"unstack(b3,b5)",
			"put-down(b3)",
			"unstack(b5,b4)",
			"put-down(b5)",
			"nop()",
			"nop()",
			"unstack(b4,b1)",
			"stack(b4,b2)",
			"nop()",
			"nop()",
			"nop()",
			"pick-up(b1)",
			"stack(b1,b4)",
			"nop()",
			"nop()",
			"nop()",
			"pick-up(b3)",
			"stack(b3,b1)",
		],
		method_library=method_library,
		method_trace=[
			{"method_name": "m_do_put_on_domain_2", "task_args": ["b4", "b2"]},
			{"method_name": "m_do_clear_domain_2", "task_args": ["b4"]},
			{"method_name": "m_do_clear_domain_2", "task_args": ["b5"]},
			{"method_name": "m_do_clear_domain_2", "task_args": ["b3"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b2"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b2"]},
			{"method_name": "m_do_on_table_domain_2", "task_args": ["b2"]},
			{"method_name": "m_do_move_domain_2", "task_args": ["b4", "b2"]},
			{"method_name": "m_do_put_on_domain_2", "task_args": ["b1", "b4"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b1"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b4"]},
			{"method_name": "m_do_on_table_domain_2", "task_args": ["b4"]},
			{"method_name": "m_do_move_domain_1", "task_args": ["b1", "b4"]},
			{"method_name": "m_do_put_on_domain_2", "task_args": ["b3", "b1"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b3"]},
			{"method_name": "m_do_clear_domain_1", "task_args": ["b1"]},
			{"method_name": "m_do_on_table_domain_2", "task_args": ["b1"]},
			{"method_name": "m_do_move_domain_1", "task_args": ["b3", "b1"]},
		],
	)

	assert plan_text is not None
	assert plan_text.startswith("==>\n")
	assert "\n0 nop\n" in f"\n{plan_text}"
	assert "\nroot " in plan_text
	assert "-> m1_do_put_on" in plan_text


def test_render_supported_hierarchical_plan_can_bridge_official_root_tasks(tmp_path):
	problem_file = tmp_path / "problem.hddl"
	problem_file.write_text(
		"""
(define (problem bridge-problem)
  (:domain BRIDGE)
  (:objects a b - block)
  (:htn
    :tasks (and
      (t1 (do_put_on a b))
    )
    :ordering (and)
  )
  (:init)
  (:goal (and))
)
""".strip()
		+ "\n",
	)
	verifier = IPCPlanVerifier()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("make_on", ("X", "Y"), False, ("on",)),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_make_on_direct",
				task_name="make_on",
				parameters=("X", "Y"),
				context=(),
				subtasks=(
					HTNMethodStep(
						step_id="s1",
						task_name="stack",
						args=("X", "Y"),
						kind="primitive",
						action_name="stack",
						literal=HTNLiteral("on", ("X", "Y"), True, None),
					),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[HTNLiteral("on", ("a", "b"), True, None)],
		target_task_bindings=[HTNTargetTaskBinding("on(a, b)", "make_on")],
	)

	plan_text = verifier._render_supported_hierarchical_plan(
		domain_file=tmp_path / "domain.hddl",
		problem_file=problem_file,
		action_path=["stack(a,b)"],
		method_library=method_library,
		method_trace=[{"method_name": "m_make_on_direct", "task_args": ["a", "b"]}],
		root_task_bridges=[
			{
				"source_task_name": "do_put_on",
				"generated_task_name": "make_on",
				"method_name": "m_verify_bridge_do_put_on_make_on",
			},
		],
	)

	assert plan_text is not None
	assert "root " in plan_text
	assert "do_put_on a b -> m_verify_bridge_do_put_on_make_on" in plan_text
	assert "make_on a b -> m_make_on_direct 0" in plan_text
