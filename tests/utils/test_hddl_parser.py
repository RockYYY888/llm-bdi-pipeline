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
