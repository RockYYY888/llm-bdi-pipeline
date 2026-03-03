"""
Stage 4 PANDA planning data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from stage3_method_synthesis.htn_schema import HTNLiteral


@dataclass(frozen=True)
class PANDAPlanStep:
	"""One primitive step in a PANDA-generated executable plan."""

	task_name: str
	action_name: str
	args: Tuple[str, ...] = ()
	source_line: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"task_name": self.task_name,
			"action_name": self.action_name,
			"args": list(self.args),
			"source_line": self.source_line,
		}


@dataclass
class PANDAPlanResult:
	"""The Stage 4 output: a PANDA-generated primitive plan."""

	task_name: str
	task_args: Tuple[str, ...]
	target_literal: Optional[HTNLiteral]
	steps: List[PANDAPlanStep] = field(default_factory=list)
	domain_hddl: str = ""
	problem_hddl: str = ""
	parser_stdout: str = ""
	parser_stderr: str = ""
	grounder_stdout: str = ""
	grounder_stderr: str = ""
	engine_stdout: str = ""
	engine_stderr: str = ""
	raw_plan: str = ""
	actual_plan: str = ""
	work_dir: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"task_name": self.task_name,
			"task_args": list(self.task_args),
			"target_literal": self.target_literal.to_dict() if self.target_literal else None,
			"steps": [step.to_dict() for step in self.steps],
			"domain_hddl": self.domain_hddl,
			"problem_hddl": self.problem_hddl,
			"parser_stdout": self.parser_stdout,
			"parser_stderr": self.parser_stderr,
			"grounder_stdout": self.grounder_stdout,
			"grounder_stderr": self.grounder_stderr,
			"engine_stdout": self.engine_stdout,
			"engine_stderr": self.engine_stderr,
			"raw_plan": self.raw_plan,
			"actual_plan": self.actual_plan,
			"work_dir": self.work_dir,
		}
