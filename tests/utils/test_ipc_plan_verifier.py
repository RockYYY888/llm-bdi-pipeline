from __future__ import annotations

import os
import subprocess
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


def test_verifier_prefers_panda_pi_bin_over_path(tmp_path, monkeypatch):
	old_dir = tmp_path / "old_bin"
	old_dir.mkdir()
	old_parser = old_dir / "pandaPIparser"
	old_parser.write_text("#!/bin/sh\nexit 0\n")
	old_parser.chmod(0o755)

	new_dir = tmp_path / "new_bin"
	new_dir.mkdir()
	new_parser = new_dir / "pandaPIparser"
	new_parser.write_text("#!/bin/sh\nexit 0\n")
	new_parser.chmod(0o755)

	monkeypatch.setenv("PATH", str(old_dir))
	monkeypatch.setenv("PANDA_PI_BIN", str(new_dir))

	verifier = IPCPlanVerifier()

	assert verifier._resolve_command_head("pandaPIparser") == str(new_parser)


def test_planning_toolchain_available_discovers_all_tools_from_panda_pi_bin(tmp_path, monkeypatch):
	parser_dir = tmp_path / "bin"
	parser_dir.mkdir()
	for tool_name in ("pandaPIparser", "pandaPIgrounder", "pandaPIengine"):
		tool_path = parser_dir / tool_name
		tool_path.write_text("#!/bin/sh\nexit 0\n")
		tool_path.chmod(0o755)

	monkeypatch.setenv("PANDA_PI_BIN", str(parser_dir))
	monkeypatch.setenv("PATH", os.getenv("PATH", ""))

	verifier = IPCPlanVerifier()

	assert verifier.planning_toolchain_available() is True


def test_verify_planned_hierarchical_plan_converts_planner_output_before_verifying(
	tmp_path,
	monkeypatch,
):
	domain_file = tmp_path / "domain.hddl"
	problem_file = tmp_path / "problem.hddl"
	domain_file.write_text("(define (domain TEST))\n")
	problem_file.write_text("(define (problem test) (:domain TEST))\n")

	converted_plan = "\n".join(
		[
			"==>",
			"0 move a b",
			"root 1",
			"1 do_move a b -> m_do_move",
		],
	) + "\n"
	raw_plan = "raw planner output\n"
	verify_stdout = "\n".join(
		[
			"Plan is executable: true",
			"Plan verification result: true",
		],
	)

	def fake_run(command, text, capture_output, check):
		if command[1:2] == ["-c"]:
			Path(command[3]).write_text(converted_plan)
			return subprocess.CompletedProcess(command, 0, "", "")
		if command[1:2] == ["-V"]:
			assert Path(command[4]).read_text() == converted_plan
			return subprocess.CompletedProcess(command, 0, verify_stdout, "")
		if command[0] == "/fake/parser":
			return subprocess.CompletedProcess(command, 0, "", "")
		if command[0] == "/fake/grounder":
			return subprocess.CompletedProcess(command, 0, "", "")
		if command[0] == "/fake/engine":
			return subprocess.CompletedProcess(command, 0, raw_plan, "")
		raise AssertionError(f"unexpected command: {command}")

	verifier = IPCPlanVerifier()
	monkeypatch.setattr(verifier, "_resolve_command_head", lambda command: f"/fake/{command[7:].lower()}" if command.startswith("pandaPI") else None)
	monkeypatch.setattr("utils.ipc_plan_verifier.subprocess.run", fake_run)

	result = verifier.verify_planned_hierarchical_plan(
		domain_file=domain_file,
		problem_file=problem_file,
		output_dir=tmp_path,
	)

	assert result.tool_available is True
	assert result.plan_kind == "hierarchical"
	assert result.verification_result is True
	assert result.primitive_plan_executable is True
	assert Path(result.plan_file).read_text() == converted_plan


def test_verify_plan_text_verifies_authoritative_hierarchical_plan_directly(tmp_path, monkeypatch):
	domain_file = tmp_path / "domain.hddl"
	problem_file = tmp_path / "problem.hddl"
	domain_file.write_text("(define (domain TEST))\n")
	problem_file.write_text("(define (problem test) (:domain TEST))\n")
	plan_text = "\n".join(
		[
			"==>",
			"0 move a b",
			"root 1",
			"1 do_move a b -> m_do_move 0",
		],
	) + "\n"
	verify_stdout = "\n".join(
		[
			"Plan is executable: true",
			"Plan verification result: true",
		],
	)

	def fake_run(command, text, capture_output, check):
		assert command[1:2] == ["-V"]
		assert Path(command[4]).read_text() == plan_text
		return subprocess.CompletedProcess(command, 0, verify_stdout, "")

	verifier = IPCPlanVerifier()
	monkeypatch.setattr(verifier, "_resolve_command_head", lambda command: "/fake/parser")
	monkeypatch.setattr("utils.ipc_plan_verifier.subprocess.run", fake_run)

	result = verifier.verify_plan_text(
		domain_file=domain_file,
		problem_file=problem_file,
		plan_text=plan_text,
		output_dir=tmp_path,
		plan_kind="hierarchical",
		build_warning="authoritative guided plan",
	)

	assert result.tool_available is True
	assert result.plan_kind == "hierarchical"
	assert result.verification_result is True
	assert result.primitive_plan_executable is True
	assert result.build_warning == "authoritative guided plan"
	assert Path(result.plan_file).read_text() == plan_text


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


def test_default_task_args_uses_leading_method_parameters_when_task_args_omitted():
	method = HTNMethod(
		method_name="m_clear_top_put_down",
		task_name="clear_top",
		parameters=("TARGET", "BLOCKER"),
	)
	task_lookup = {
		"clear_top": HTNTask("clear_top", ("B",), False, ("clear",)),
	}

	assert IPCPlanVerifier._default_task_args(method, task_lookup) == ("TARGET",)


def test_default_task_args_keeps_zero_arity_task_distinct_from_aux_parameters():
	method = HTNMethod(
		method_name="m_helper_handempty_constructive",
		task_name="helper_handempty",
		parameters=("AUX_BLOCK1",),
	)
	task_lookup = {
		"helper_handempty": HTNTask("helper_handempty", (), False, ("handempty",)),
	}

	assert IPCPlanVerifier._default_task_args(method, task_lookup) == ()


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


def test_render_supported_hierarchical_plan_skips_mismatched_noop_trace_artifacts(tmp_path):
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
  (:action nop
    :parameters ()
    :precondition (and)
    :effect (and)
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
  (:objects a b c d - block)
  (:htn
    :tasks (and
      (t1 (assemble a b))
      (t2 (assemble c d))
    )
    :ordering (and
      (< t1 t2)
    )
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
				method_name="m_assemble_done",
				task_name="assemble",
				parameters=("X", "Y"),
				context=(
					HTNLiteral("linked", ("X", "Y"), True),
				),
				subtasks=(
					HTNMethodStep("s1", "nop", (), "primitive", action_name="nop"),
				),
				ordering=(),
			),
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
			"nop()",
			"polish(c)",
			"attach(c,d)",
		],
		method_library=method_library,
		method_trace=[
			{"method_name": "m_assemble_direct", "task_args": ["a", "b"]},
			{"method_name": "m_prepare_polish", "task_args": ["a"]},
			{"method_name": "m_assemble_done", "task_args": ["a", "b"]},
			{"method_name": "m_assemble_direct", "task_args": ["c", "d"]},
			{"method_name": "m_prepare_polish", "task_args": ["c"]},
		],
	)

	assert plan_text is not None
	assert "\n0 polish a\n" in f"\n{plan_text}"
	assert "\n1 attach a b\n" in f"\n{plan_text}"
	assert "\n2 polish c\n" in f"\n{plan_text}"
	assert "\n3 attach c d\n" in f"\n{plan_text}"
	assert "nop" not in plan_text
	assert "assemble a b -> m_assemble_direct" in plan_text
	assert "assemble c d -> m_assemble_direct" in plan_text


def test_render_supported_hierarchical_plan_accepts_unordered_root_task_reordering(tmp_path):
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
  (:objects a b c d - block)
  (:htn
    :tasks (and
      (t1 (assemble a b))
      (t2 (assemble c d))
    )
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
			"polish(c)",
			"attach(c,d)",
			"polish(a)",
			"attach(a,b)",
		],
		method_library=method_library,
		method_trace=[
			{"method_name": "m_assemble_direct", "task_args": ["c", "d"]},
			{"method_name": "m_prepare_polish", "task_args": ["c"]},
			{"method_name": "m_assemble_direct", "task_args": ["a", "b"]},
			{"method_name": "m_prepare_polish", "task_args": ["a"]},
		],
	)

	assert plan_text is not None
	assert "assemble c d -> m_assemble_direct" in plan_text
	assert "assemble a b -> m_assemble_direct" in plan_text
	assert plan_text.index("assemble c d -> m_assemble_direct") < plan_text.index(
		"assemble a b -> m_assemble_direct",
	)


def test_render_supported_hierarchical_plan_relaxes_order_for_query_root_alias_tasks(tmp_path):
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
(define (problem render-query-root-plan)
  (:domain TEST)
  (:objects a b c d - block)
  (:htn
    :tasks (and
      (t1 (assemble a b))
      (t2 (assemble c d))
    )
    :ordering (and
      (< t1 t2)
    )
  )
  (:init)
  (:goal (and))
)
""".strip()
		+ "\n",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("query_root_1_assemble", ("X", "Y"), False, source_name="assemble"),
			HTNTask("query_root_2_assemble", ("X", "Y"), False, source_name="assemble"),
			HTNTask("prepare", ("X",), False),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_query_root_1_assemble_constructive_1",
				task_name="query_root_1_assemble",
				parameters=("X", "Y"),
				subtasks=(
					HTNMethodStep("s1", "prepare", ("X",), "compound"),
					HTNMethodStep("s2", "attach", ("X", "Y"), "primitive", action_name="attach"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_query_root_2_assemble_constructive_1",
				task_name="query_root_2_assemble",
				parameters=("X", "Y"),
				subtasks=(
					HTNMethodStep("s1", "prepare", ("X",), "compound"),
					HTNMethodStep("s2", "attach", ("X", "Y"), "primitive", action_name="attach"),
				),
				ordering=(("s1", "s2"),),
				origin="llm",
			),
			HTNMethod(
				method_name="m_prepare_polish",
				task_name="prepare",
				parameters=("X",),
				subtasks=(
					HTNMethodStep("s1", "polish", ("X",), "primitive", action_name="polish"),
				),
				ordering=(),
				origin="llm",
			),
		],
		target_literals=[],
		target_task_bindings=[],
	)

	plan_text = verifier._render_supported_hierarchical_plan(
		domain_file=domain_file,
		problem_file=problem_file,
		action_path=[
			"polish(c)",
			"attach(c,d)",
			"polish(a)",
			"attach(a,b)",
		],
		method_library=method_library,
		method_trace=[
			{"method_name": "m_query_root_2_assemble_constructive_1", "task_args": ["c", "d"]},
			{"method_name": "m_prepare_polish", "task_args": ["c"]},
			{"method_name": "m_query_root_1_assemble_constructive_1", "task_args": ["a", "b"]},
			{"method_name": "m_prepare_polish", "task_args": ["a"]},
		],
	)

	assert plan_text is not None
	assert "assemble c d -> m_query_root_2_assemble_constructive_1" in plan_text
	assert "assemble a b -> m_query_root_1_assemble_constructive_1" in plan_text
	assert plan_text.index("assemble c d -> m_query_root_2_assemble_constructive_1") < plan_text.index(
		"assemble a b -> m_query_root_1_assemble_constructive_1",
	)


def test_render_supported_hierarchical_plan_accepts_reordered_leading_compound_prefix(tmp_path):
	verifier = IPCPlanVerifier()
	domain_file = tmp_path / "domain.hddl"
	problem_file = tmp_path / "problem.hddl"
	domain_file.write_text(
		"""
(define (domain TEST)
  (:requirements :hierarchy :typing)
  (:types rover waypoint store)
  (:predicates (sampled ?r - rover ?w - waypoint))
  (:task collect_sample
    :parameters (?r - rover ?w - waypoint ?s - store)
  )
  (:task helper_at
    :parameters (?r - rover ?w - waypoint)
  )
  (:task helper_empty
    :parameters (?s - store)
  )
  (:action move
    :parameters (?r - rover ?w - waypoint)
    :precondition (and)
    :effect (and)
  )
  (:action sample
    :parameters (?r - rover ?s - store ?w - waypoint)
    :precondition (and)
    :effect (and (sampled ?r ?w))
  )
)
""".strip()
		+ "\n",
	)
	problem_file.write_text(
		"""
(define (problem render-reordered-compound-prefix)
  (:domain TEST)
  (:objects rover1 - rover waypoint1 - waypoint store1 - store)
  (:htn
    :tasks (and
      (t1 (collect_sample rover1 waypoint1 store1))
    )
  )
  (:init)
  (:goal (and))
)
""".strip()
		+ "\n",
	)
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask("collect_sample", ("ROVER", "WAYPOINT", "STORE"), False),
			HTNTask("helper_at", ("ROVER", "WAYPOINT"), False),
			HTNTask("helper_empty", ("STORE",), False),
		],
		primitive_tasks=[],
		methods=[
			HTNMethod(
				method_name="m_collect_sample",
				task_name="collect_sample",
				parameters=("ROVER", "WAYPOINT", "STORE"),
				subtasks=(
					HTNMethodStep("s1", "helper_at", ("ROVER", "WAYPOINT"), "compound"),
					HTNMethodStep("s2", "helper_empty", ("STORE",), "compound"),
					HTNMethodStep(
						"s3",
						"sample",
						("ROVER", "STORE", "WAYPOINT"),
						"primitive",
						action_name="sample",
					),
				),
				ordering=(("s1", "s3"), ("s2", "s3")),
			),
			HTNMethod(
				method_name="m_helper_at_move",
				task_name="helper_at",
				parameters=("ROVER", "WAYPOINT"),
				subtasks=(
					HTNMethodStep(
						"s1",
						"move",
						("ROVER", "WAYPOINT"),
						"primitive",
						action_name="move",
					),
				),
				ordering=(),
			),
			HTNMethod(
				method_name="m_helper_empty_noop",
				task_name="helper_empty",
				parameters=("STORE",),
				subtasks=(),
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
			"move(rover1,waypoint1)",
			"sample(rover1,store1,waypoint1)",
		],
		method_library=method_library,
		method_trace=[
			{
				"method_name": "m_collect_sample",
				"task_args": ["rover1", "waypoint1", "store1"],
			},
			{
				"method_name": "m_helper_empty_noop",
				"task_args": ["store1"],
			},
			{
				"method_name": "m_helper_at_move",
				"task_args": ["rover1", "waypoint1"],
			},
		],
	)

	assert plan_text is not None
	assert "collect_sample rover1 waypoint1 store1 -> m_collect_sample" in plan_text
	assert "helper_empty store1 -> m_helper_empty_noop" in plan_text
	assert "helper_at rover1 waypoint1 -> m_helper_at_move" in plan_text
	assert "0 move rover1 waypoint1" in plan_text
	assert "1 sample rover1 store1 waypoint1" in plan_text
	assert plan_text.index("helper_at rover1 waypoint1 -> m_helper_at_move") < plan_text.index(
		"helper_empty store1 -> m_helper_empty_noop",
	)


def test_unify_arguments_allows_schematic_binding_to_refine_into_ground_object():
	bindings = IPCPlanVerifier._unify_arguments(
		("ARG1", "ARG2"),
		("?direction1", "?mode1"),
		{},
		("ARG1", "ARG2"),
	)

	refined = IPCPlanVerifier._unify_arguments(
		("ARG1", "ARG2"),
		("star5", "image1"),
		bindings,
		("ARG1", "ARG2"),
	)

	assert refined == {"ARG1": "star5", "ARG2": "image1"}


def test_refine_task_args_uses_method_bindings_for_schematic_trace_args():
	method = HTNMethod(
		method_name="m_observe",
		task_name="do_observation",
		parameters=("ARG1", "ARG2"),
		task_args=("ARG1", "ARG2"),
	)

	refined = IPCPlanVerifier._refine_task_args_from_method_bindings(
		task_args=("?direction1", "?mode1"),
		method=method,
		task_lookup={},
		bindings={"ARG1": "star5", "ARG2": "image1"},
	)

	assert refined == ("star5", "image1")
