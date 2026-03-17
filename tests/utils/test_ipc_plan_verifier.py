from __future__ import annotations

import os
import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage3_method_synthesis.htn_schema import (
	HTNLiteral,
	HTNMethod,
	HTNMethodLibrary,
	HTNMethodStep,
	HTNTargetTaskBinding,
	HTNTask,
)
from utils.ipc_plan_verifier import IPCPlanVerifier


def test_tool_available_discovers_parser_from_panda_pi_bin(tmp_path, monkeypatch):
	parser_dir = tmp_path / "bin"
	parser_dir.mkdir()
	parser_path = parser_dir / "pandaPIparser"
	parser_path.write_text("#!/bin/sh\nexit 0\n")
	parser_path.chmod(0o755)

	monkeypatch.setenv("PANDA_PI_BIN", str(parser_dir))
	monkeypatch.setenv("PATH", os.getenv("PATH", ""))

	verifier = IPCPlanVerifier()

	assert verifier.tool_available() is True


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


def test_render_supported_hierarchical_plan_contains_root_and_decompositions(tmp_path):
	verifier = IPCPlanVerifier()
	domain_file = tmp_path / "domain.hddl"
	problem_file = tmp_path / "problem.hddl"
	domain_file.write_text(
		"""
(define (domain TEST)
  (:requirements :hierarchy :typing)
  (:types block)
  (:predicates (ready ?x - block) (linked ?x - block ?y - block))
  (:task assemble
    :parameters (?x - block ?y - block)
  )
  (:task prepare
    :parameters (?x - block)
  )
  (:method noop
    :parameters ()
    :task (assemble ?x ?y)
    :subtasks ()
  )
  (:action polish
    :parameters (?x - block)
    :precondition (and)
    :effect (and (ready ?x))
  )
  (:action attach
    :parameters (?x - block ?y - block)
    :precondition (and (ready ?x))
    :effect (and (linked ?x ?y))
  )
)
""".strip()
		+ "\n",
	)
	problem_file.write_text(
		"""
(define (problem render-plan)
  (:domain TEST)
  (:objects a b - block)
  (:htn
    :tasks (and
      (t1 (assemble a b))
    )
    :ordering (and)
  )
  (:init)
  (:goal (and))
)
""".strip()
		+ "\n",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("assemble", ("X", "Y"), False),
			HTNTask("prepare", ("X",), False),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_assemble_direct",
				task_name="assemble",
				parameters=("X", "Y"),
				subtasks=(
					HTNMethodStep("s1", "prepare", ("X",), "compound"),
					HTNMethodStep("s2", "attach", ("X", "Y"), "primitive", action_name="attach"),
				),
				ordering=(("s1", "s2"),),
			),
			HTNMethod(
				method_name="m_prepare_polish",
				task_name="prepare",
				parameters=("X",),
				subtasks=(
					HTNMethodStep("s1", "polish", ("X",), "primitive", action_name="polish"),
				),
				ordering=(),
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_text = verifier._render_supported_hierarchical_plan(
		domain_file=domain_file,
		problem_file=problem_file,
		action_path=[
			"polish(a)",
			"attach(a,b)",
		],
		method_library=method_library,
		method_trace=[
			{"method_name": "m_assemble_direct", "task_args": ["a", "b"]},
			{"method_name": "m_prepare_polish", "task_args": ["a"]},
		],
	)

	assert plan_text is not None
	assert plan_text.startswith("==>\n")
	assert "\n0 polish a\n" in f"\n{plan_text}"
	assert "\nroot " in plan_text
	assert "assemble a b -> m_assemble_direct" in plan_text
	assert "prepare a -> m_prepare_polish" in plan_text
