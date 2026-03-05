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

    assert domain.name == "blocksworld"
    assert ":hierarchy" in domain.requirements
    assert "place_on" in task_names
    assert "m_hold_block_from_stack" in method_names
    assert "put-on-block" in action_names


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
        / "example_problem.hddl"
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
