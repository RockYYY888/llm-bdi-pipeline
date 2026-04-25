"""
Public HTN evaluation entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from execution_logging.execution_logger import ExecutionLogger

from .context import HTNEvaluationContext
from .problem_root_evaluator import HTNProblemRootEvaluator
from .result_tables import PRIMARY_HTN_PLANNER_ID, SINGLE_PLANNER_MODE


class HTNEvaluationPipeline:
	"""Public entrypoint for planner-based HTN benchmark evaluation."""

	def __init__(
		self,
		*,
		domain_file: str,
		problem_file: str,
	) -> None:
		self._context = HTNEvaluationContext(
			domain_file=domain_file,
			problem_file=problem_file,
		)
		self._htn_problem_root_evaluator_instance: Optional[HTNProblemRootEvaluator] = None

	@property
	def context(self) -> HTNEvaluationContext:
		return self._context

	@property
	def config(self):
		return self._context.config

	@config.setter
	def config(self, value) -> None:
		self._context.config = value

	@property
	def domain(self):
		return self._context.domain

	@property
	def problem(self):
		return self._context.problem

	@property
	def domain_file(self) -> str:
		return self._context.domain_file

	@property
	def problem_file(self) -> str:
		return self._context.problem_file

	@property
	def output_dir(self) -> Optional[Path]:
		return self._context.output_dir

	@output_dir.setter
	def output_dir(self, value: Optional[str | Path]) -> None:
		self._context.output_dir = Path(value).resolve() if value is not None else None

	@property
	def logger(self) -> ExecutionLogger:
		return self._context.logger

	@logger.setter
	def logger(self, value: ExecutionLogger) -> None:
		self._context.logger = value

	def _official_problem_root_structure_analysis(self):
		return self._context._official_problem_root_structure_analysis()

	def _official_problem_root_planning_timeout_seconds(
		self,
		timeout_seconds: Optional[float] = None,
	) -> float:
		return self._context._official_problem_root_planning_timeout_seconds(timeout_seconds)

	def _merge_primary_planner_output_dir(self, source_root: Path) -> None:
		self._context._merge_primary_planner_output_dir(source_root)

	def _execute_primary_htn_planner(
		self,
		method_library: Any = None,
	) -> Dict[str, Any]:
		"""Run the supported lifted_panda_sat primary HTN baseline."""
		return self.execute_problem_root_evaluation(
			method_library=method_library,
			evaluation_mode=SINGLE_PLANNER_MODE,
			planner_id=PRIMARY_HTN_PLANNER_ID,
		)

	def execute_problem_root_evaluation(
		self,
		*,
		method_library: Any = None,
		evaluation_mode: str = SINGLE_PLANNER_MODE,
		planner_id: Optional[str] = PRIMARY_HTN_PLANNER_ID,
	) -> Dict[str, Any]:
		_ = method_library
		evaluator = self._htn_problem_root_evaluator_instance
		if evaluator is None:
			evaluator = HTNProblemRootEvaluator(self._context)
			self._htn_problem_root_evaluator_instance = evaluator
		return evaluator.execute_problem_root_evaluation(
			method_library=method_library,
			evaluation_mode=evaluation_mode,
			planner_id=planner_id,
		)

	@staticmethod
	def _close_planner_queue(result_queue: Any) -> None:
		HTNProblemRootEvaluator.close_planner_queue(result_queue)
