"""
Focused tests for the HDDL domain parser.
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.hddl_parser import HDDLParser


def test_blocksworld_hddl_domain_exposes_tasks_methods_and_actions():
    domain_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "blocksworld"
        / "domain.hddl"
    )

    domain = HDDLParser.parse_domain(str(domain_path))

    task_names = {task.name for task in domain.tasks}
    method_names = {method.name for method in domain.methods}
    action_names = {action.name for action in domain.actions}

    assert domain.name.lower() == "blocks"
    assert ":hierarchy" in domain.requirements
    assert "do_put_on" in task_names
    assert "m5_do_move" in method_names
    assert "stack" in action_names


def test_marsrover_hddl_domain_exposes_tasks_methods_and_actions():
    domain_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "marsrover"
        / "domain.hddl"
    )

    domain = HDDLParser.parse_domain(str(domain_path))

    task_names = {task.name for task in domain.tasks}
    method_names = {method.name for method in domain.methods}
    action_names = {action.name for action in domain.actions}

    assert domain.name == "rover"
    assert ":method-preconditions" in domain.requirements
    assert "navigate_abs" in task_names
    assert "m-navigate_abs-1" in method_names
    assert "navigate" in action_names


def test_marsrover_problem_parser_extracts_objects_init_and_htn_tasks():
    problem_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "marsrover"
        / "problems"
        / "pfile06.hddl"
    )

    problem = HDDLParser.parse_problem(str(problem_path))

    assert problem.name == "roverprob2312"
    assert problem.domain_name.lower() == "rover"
    assert "rover0" in problem.objects
    assert problem.object_types["rover0"] == "rover"
    assert any(
        fact.predicate == "at" and fact.args == ["rover0", "waypoint1"]
        for fact in problem.init_facts
    )
    assert any(
        task.task_name == "get_rock_data" and task.args == ["waypoint2"]
        for task in problem.htn_tasks
    )


def test_blocksworld_problem_parser_extracts_ordered_subtasks_as_htn_tasks():
    problem_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "blocksworld"
        / "problems"
        / "p01.hddl"
    )

    problem = HDDLParser.parse_problem(str(problem_path))

    assert any(
        task.task_name == "do_put_on" and task.args == ["b4", "b2"]
        for task in problem.htn_tasks
    )
    assert any(
        task.task_name == "do_put_on" and task.args == ["b1", "b4"]
        for task in problem.htn_tasks
    )
    assert problem.htn_ordered is True
    assert problem.htn_ordering == []
    assert problem.htn_tasks[0].label == "task1"


def test_satellite_hddl_domain_exposes_tasks_methods_and_actions():
    domain_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "satellite"
        / "domain.hddl"
    )

    domain = HDDLParser.parse_domain(str(domain_path))

    task_names = {task.name for task in domain.tasks}
    method_names = {method.name for method in domain.methods}
    action_names = {action.name for action in domain.actions}

    assert domain.name == "satellite2"
    assert "do_observation" in task_names
    assert "method0" in method_names
    assert "take_image" in action_names


def test_satellite_problem_parser_extracts_labelled_subtasks_as_htn_tasks():
    problem_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "satellite"
        / "problems"
        / "1obs-1sat-1mod.hddl"
    )

    problem = HDDLParser.parse_problem(str(problem_path))

    assert any(
        task.task_name == "do_observation"
        and task.args == ["Phenomenon4", "thermograph0"]
        for task in problem.htn_tasks
    )
    assert problem.htn_parameter_types == {}


def test_satellite_problem_parser_extracts_htn_parameters():
    problem_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "satellite"
        / "problems"
        / "1obs-2sat-1mod.hddl"
    )

    problem = HDDLParser.parse_problem(str(problem_path))

    assert problem.htn_parameter_types == {
        "?direction1": "image_direction",
        "?mode1": "mode",
    }
    assert [(task.label, task.task_name, task.args) for task in problem.htn_tasks] == [
        ("task0", "do_observation", ["?direction1", "?mode1"]),
    ]


def test_problem_parser_extracts_explicit_root_ordering_edges(tmp_path):
    problem_path = tmp_path / "ordered_problem.hddl"
    problem_path.write_text(
        """
(define (problem demo-problem)
 (:domain demo-domain)
 (:objects a b - object)
 (:htn
  :tasks (and
   (t1 (deliver a))
   (t2 (deliver b))
  )
  :ordering (and
   (< t1 t2)
  )
 )
 (:init)
)
        """.strip()
    )

    problem = HDDLParser.parse_problem(str(problem_path))

    assert problem.htn_ordered is False
    assert problem.htn_ordering == [("t1", "t2")]
    assert [(task.label, task.task_name, task.args) for task in problem.htn_tasks] == [
        ("t1", "deliver", ["a"]),
        ("t2", "deliver", ["b"]),
    ]


def test_transport_hddl_domain_exposes_tasks_methods_and_actions():
    domain_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "domains"
        / "transport"
        / "domain.hddl"
    )

    domain = HDDLParser.parse_domain(str(domain_path))

    task_names = {task.name for task in domain.tasks}
    method_names = {method.name for method in domain.methods}
    action_names = {action.name for action in domain.actions}

    assert domain.name == "transport"
    assert "deliver" in task_names
    assert "m-deliver" in method_names
    assert "drive" in action_names
