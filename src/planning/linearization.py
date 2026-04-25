"""
Lifted Linear wrapper for plan solve official problem-root planning.

This adapter mirrors the IPC 2023 Lifted Linear config-2 flow for partial-order
HTN problems:
1. Linearize the original partial-order HDDL instance into a total-order HDDL
   instance.
2. Solve the linearized instance with lifted PANDA SAT search.
3. Decode the tokenized plan text back into the original symbol space before
   handing it to the official verifier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from planning.official_benchmark import (
	OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID,
	OFFICIAL_LIFTED_LINEAR_SOLVER_ID,
)
from planning.panda_sat import PANDAPlanner, PANDAPlanningError
from planning.plan_models import PANDAPlanResult
from planning.process_capture import run_subprocess_to_files


class LiftedLinearPlanner:
	"""Run the IPC 2023 Lifted Linear config-2 workflow locally."""

	SOLVER_ID = OFFICIAL_LIFTED_LINEAR_SOLVER_ID
	INNER_SOLVER_ID = OFFICIAL_LIFTED_LINEAR_INNER_SOLVER_ID
	_TOKEN_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
		("LA_", "<"),
		("RA_", ">"),
		("LB_", "["),
		("RB_", "]"),
		("BAR_", "|"),
		("SEM_", ";"),
		("COM_", ","),
		("PLUS_", "+"),
		("MINUS_", "-"),
		("EXCLAMATION_", "!"),
		("US_", "_"),
	)

	def __init__(
		self,
		workspace: Optional[str | Path] = None,
		panda_planner: Optional[PANDAPlanner] = None,
	) -> None:
		self.workspace = Path(workspace).resolve() if workspace else None
		self.panda_planner = panda_planner or PANDAPlanner(
			workspace=self.workspace,
			parser_cmd=self._preferred_parser_command(),
			grounder_cmd=self._preferred_grounder_command(),
			engine_cmd=self._preferred_engine_command(),
		)

	def toolchain_available(self) -> bool:
		return (
			self._resolve_linearizer_binary() is not None
			and self.panda_planner.toolchain_available()
		)

	def plan_hddl_files(
		self,
		*,
		domain: Any,
		domain_file: str | Path,
		problem_file: str | Path,
		task_name: str,
		transition_name: str,
		task_args: Optional[Sequence[str]] = None,
		timeout_seconds: Optional[float] = None,
	) -> PANDAPlanResult:
		linearizer = self._resolve_linearizer_binary()
		if linearizer is None:
			raise PANDAPlanningError(
				"Lifted Linear toolchain is unavailable.",
				metadata={
					"backend": self.SOLVER_ID,
					"engine_attempts": [
						{
							"solver_id": self.SOLVER_ID,
							"mode": self.INNER_SOLVER_ID,
							"status": "skipped",
							"reason": "linearizer_unavailable",
						},
					],
				},
			)

		total_start = time.perf_counter()
		linearization = self.linearize_hddl_files(
			domain_file=domain_file,
			problem_file=problem_file,
			transition_name=transition_name,
			timeout_seconds=timeout_seconds,
		)
		inner_result = self.plan_linearized_hddl_files(
			domain=domain,
			linearized_domain_file=linearization["linearized_domain_file"],
			linearized_problem_file=linearization["linearized_problem_file"],
			task_name=task_name,
			transition_name=transition_name,
			task_args=tuple(task_args or ()),
			timeout_seconds=self._remaining_timeout_seconds(total_start, timeout_seconds),
			solver_configs=(self.panda_planner._solver_config_by_id(self.INNER_SOLVER_ID),),
			reported_solver_id=self.SOLVER_ID,
			reported_engine_mode=self.INNER_SOLVER_ID,
			linearization_metadata=linearization,
		)
		inner_result.timing_profile = {
			**dict(inner_result.timing_profile or {}),
			"total_seconds": time.perf_counter() - total_start,
		}
		return inner_result

	def linearize_hddl_files(
		self,
		*,
		domain_file: str | Path,
		problem_file: str | Path,
		transition_name: str,
		timeout_seconds: Optional[float] = None,
	) -> Dict[str, Any]:
		linearizer = self._resolve_linearizer_binary()
		if linearizer is None:
			raise PANDAPlanningError(
				"Lifted Linear toolchain is unavailable.",
				metadata={
					"backend": self.SOLVER_ID,
					"engine_attempts": [
						{
							"solver_id": self.SOLVER_ID,
							"mode": self.INNER_SOLVER_ID,
							"status": "skipped",
							"reason": "linearizer_unavailable",
						},
					],
				},
			)

		work_dir = self._resolve_work_dir(transition_name)
		work_dir.mkdir(parents=True, exist_ok=True)
		linearized_domain_path = work_dir / "domain.linearized.hddl"
		linearized_problem_path = work_dir / "problem.linearized.hddl"

		linearizer_start = time.perf_counter()
		linearizer_result = self._run_subprocess(
			[
				str(linearizer),
				str(Path(domain_file).resolve()),
				str(Path(problem_file).resolve()),
				str(linearized_domain_path),
				str(linearized_problem_path),
			],
			work_dir,
			timeout_seconds=timeout_seconds,
			output_label="linearizer",
		)
		if linearizer_result["returncode"] != 0:
			raise PANDAPlanningError(
				"Lifted Linear linearization failed.",
				metadata={
					"backend": self.SOLVER_ID,
					"linearizer_stdout": linearizer_result["stdout"],
					"linearizer_stderr": linearizer_result["stderr"],
					"linearizer_stdout_path": linearizer_result["stdout_path"],
					"linearizer_stderr_path": linearizer_result["stderr_path"],
					"engine_attempts": [
						{
							"solver_id": self.SOLVER_ID,
							"mode": self.INNER_SOLVER_ID,
							"status": "failed",
							"reason": "linearizer_failed",
							"stdout": linearizer_result["stdout"],
							"stderr": linearizer_result["stderr"],
							"stdout_path": linearizer_result["stdout_path"],
							"stderr_path": linearizer_result["stderr_path"],
						},
					],
				},
			)
		return {
			"work_dir": str(work_dir),
			"linearized_domain_file": str(linearized_domain_path),
			"linearized_problem_file": str(linearized_problem_path),
			"linearizer_seconds": time.perf_counter() - linearizer_start,
			"linearizer_stdout": linearizer_result["stdout"],
			"linearizer_stderr": linearizer_result["stderr"],
			"linearizer_stdout_path": linearizer_result["stdout_path"],
			"linearizer_stderr_path": linearizer_result["stderr_path"],
		}

	def plan_linearized_hddl_files(
		self,
		*,
		domain: Any,
		linearized_domain_file: str | Path,
		linearized_problem_file: str | Path,
		task_name: str,
		transition_name: str,
		task_args: Optional[Sequence[str]] = None,
		timeout_seconds: Optional[float] = None,
		solver_configs: Optional[Sequence[Dict[str, Any]]] = None,
		reported_solver_id: Optional[str] = None,
		reported_engine_mode: Optional[str] = None,
		linearization_metadata: Optional[Dict[str, Any]] = None,
	) -> PANDAPlanResult:
		work_dir = self._resolve_work_dir(transition_name)
		work_dir.mkdir(parents=True, exist_ok=True)
		inner_workspace = work_dir / "inner"
		inner_planner = PANDAPlanner(
			workspace=inner_workspace,
			parser_cmd=self._preferred_parser_command(self.panda_planner.parser_cmd),
			grounder_cmd=self._preferred_grounder_command(self.panda_planner.grounder_cmd),
			engine_cmd=self._preferred_engine_command(self.panda_planner.engine_cmd),
			problem_builder=self.panda_planner.problem_builder,
		)
		inner_result = inner_planner.plan_hddl_files(
			domain=domain,
			domain_file=linearized_domain_file,
			problem_file=linearized_problem_file,
			task_name=task_name,
			transition_name=transition_name,
			task_args=tuple(task_args or ()),
			timeout_seconds=timeout_seconds,
			solver_configs=solver_configs,
			collect_all_candidates=True,
		)

		decoded_actual_plan = ""
		decoded_steps = list(inner_result.steps)
		decoded_candidates: List[Dict[str, Any]] = []
		for candidate in inner_result.solver_candidates or ():
			decoded = self._decode_solver_candidate(
				candidate=dict(candidate),
				domain=domain,
				work_dir=work_dir,
				timeout_seconds=timeout_seconds,
				reported_solver_id=reported_solver_id,
				reported_mode=reported_engine_mode,
			)
			decoded_candidates.append(decoded)
			if (
				not decoded_actual_plan
				and str(decoded.get("status") or "") == "success"
			):
				decoded_actual_plan = str(decoded.get("actual_plan_text") or "")
				decoded_steps = list(decoded.get("parsed_steps") or ())

		if not decoded_actual_plan:
			metadata = dict(linearization_metadata or {})
			raise PANDAPlanningError(
				f"Lifted Linear returned no executable primitive plan for {task_name}({', '.join(task_args or ())})",
				metadata={
					"backend": reported_solver_id or self.SOLVER_ID,
					"linearizer_stdout": metadata.get("linearizer_stdout"),
					"linearizer_stderr": metadata.get("linearizer_stderr"),
					"engine_attempts": decoded_candidates,
				},
			)

		inner_result.engine_mode = reported_engine_mode or inner_result.engine_mode
		if reported_solver_id:
			inner_result.solver_id = reported_solver_id
		inner_result.actual_plan = decoded_actual_plan
		inner_result.steps = decoded_steps
		inner_result.solver_candidates = [
			{
				key: value
				for key, value in candidate.items()
				if key not in {"actual_plan_text", "parsed_steps"}
			}
			for candidate in decoded_candidates
		]
		inner_result.work_dir = str(work_dir)
		inner_result.timing_profile = {
			**dict(inner_result.timing_profile or {}),
			"linearizer_seconds": float((linearization_metadata or {}).get("linearizer_seconds") or 0.0),
			"linearizer_stdout": (linearization_metadata or {}).get("linearizer_stdout", ""),
			"linearizer_stderr": (linearization_metadata or {}).get("linearizer_stderr", ""),
			"linearizer_stdout_path": (linearization_metadata or {}).get("linearizer_stdout_path"),
			"linearizer_stderr_path": (linearization_metadata or {}).get("linearizer_stderr_path"),
			"linearized_domain_file": str(Path(linearized_domain_file).resolve()),
			"linearized_problem_file": str(Path(linearized_problem_file).resolve()),
		}
		return inner_result

	def _decode_solver_candidate(
		self,
		*,
		candidate: Dict[str, Any],
		domain: Any,
		work_dir: Path,
		timeout_seconds: Optional[float],
		reported_solver_id: Optional[str] = None,
		reported_mode: Optional[str] = None,
	) -> Dict[str, Any]:
		if reported_solver_id:
			candidate["solver_id"] = reported_solver_id
		if reported_mode:
			candidate["mode"] = reported_mode
		raw_plan_path_value = candidate.get("raw_plan_path")
		if not raw_plan_path_value or str(candidate.get("status") or "") != "success":
			return candidate

		raw_plan_path = Path(str(raw_plan_path_value)).resolve()
		decoded_plan_path = work_dir / f"{raw_plan_path.stem}.linearizer.decoded.txt"
		converted_plan_path = work_dir / f"{raw_plan_path.stem}.linearizer.actual.txt"
		decoded_plan_path.write_text(self._decode_linearizer_tokens(raw_plan_path.read_text()))
		conversion_result = self._run_subprocess(
			[
				self.panda_planner.parser_cmd,
				"-c",
				str(decoded_plan_path),
				str(converted_plan_path),
			],
			work_dir,
			timeout_seconds=timeout_seconds,
			output_label=f"linearized_convert_{raw_plan_path.stem}",
		)
		if conversion_result["returncode"] != 0:
			raise PANDAPlanningError(
				"Lifted Linear plan conversion failed.",
				metadata={
					"backend": self.SOLVER_ID,
					"engine_attempts": [
						{
							**candidate,
							"status": "failed",
							"reason": "linearized_plan_conversion_failed",
							"stdout": conversion_result["stdout"],
							"stderr": conversion_result["stderr"],
							"stdout_path": conversion_result["stdout_path"],
							"stderr_path": conversion_result["stderr_path"],
						},
					],
				},
			)
		actual_plan_text = (
			converted_plan_path.read_text()
			if converted_plan_path.exists()
			else decoded_plan_path.read_text()
		)
		parsed_steps = self.panda_planner._parse_plan_steps(actual_plan_text, domain)
		candidate.update(
			{
				"actual_plan_path": str(converted_plan_path),
				"actual_plan_text": actual_plan_text,
				"parsed_steps": parsed_steps,
				"step_count": len(parsed_steps),
				"action_path": [
					f"{step.action_name}({', '.join(step.args)})" if step.args else step.action_name
					for step in parsed_steps
				],
				"steps": [step.to_dict() for step in parsed_steps],
				"has_hierarchical_trace": "->" in actual_plan_text,
			},
		)
		return candidate

	@classmethod
	def _decode_linearizer_tokens(cls, plan_text: str) -> str:
		decoded = str(plan_text or "")
		for source, target in cls._TOKEN_REPLACEMENTS:
			decoded = decoded.replace(source, target)
		return decoded

	def _resolve_work_dir(self, transition_name: str) -> Path:
		if self.workspace is None:
			return Path.cwd() / "tmp_lifted_linear" / transition_name
		return self.workspace / "lifted_linear" / transition_name

	@staticmethod
	def _remaining_timeout_seconds(
		total_start: float,
		timeout_seconds: Optional[float],
	) -> Optional[float]:
		if timeout_seconds is None:
			return None
		return max(timeout_seconds - (time.perf_counter() - total_start), 0.0)

	def _resolve_linearizer_binary(self) -> Optional[Path]:
		candidates: List[Path] = []
		env_bin = os.getenv("LIFTED_LINEAR_BIN")
		if env_bin:
			candidates.append(Path(env_bin).expanduser())
		for root in self._candidate_roots():
			candidates.append(root / "linearizer" / "linearizer")
		for candidate in candidates:
			if candidate.exists():
				return candidate.resolve()
		for root in self._candidate_roots():
			if self._build_linearizer(root):
				binary = root / "linearizer" / "linearizer"
				if binary.exists():
					return binary.resolve()
		return None

	def _preferred_parser_command(self, fallback: Optional[str] = None) -> str:
		resolved = self._resolve_official_binary("pandaPIparser", "pandaPIparser")
		if resolved is not None:
			return str(resolved)
		return str(fallback or "pandaPIparser")

	def _preferred_grounder_command(self, fallback: Optional[str] = None) -> str:
		resolved = self._resolve_official_binary("pandaPIgrounder", "pandaPIgrounder")
		if resolved is not None:
			return str(resolved)
		return str(fallback or "pandaPIgrounder")

	def _preferred_engine_command(self, fallback: Optional[str] = None) -> str:
		env_override = os.getenv("LIFTED_LINEAR_PANDA_ENGINE")
		if env_override:
			candidate = Path(env_override).expanduser()
			if candidate.exists():
				return str(candidate.resolve())
		resolved = self._resolve_official_binary("pandaPIengine", "build/pandaPIengine")
		if resolved is not None:
			return str(resolved)
		return str(fallback or "pandaPIengine")

	def _resolve_official_binary(self, component: str, relative_path: str) -> Optional[Path]:
		for root in self._candidate_roots():
			candidate = root / component / relative_path
			if candidate.exists() and os.access(candidate, os.X_OK):
				return candidate.resolve()
		return None

	def _candidate_roots(self) -> Tuple[Path, ...]:
		repo_root = Path(__file__).resolve().parents[2]
		roots: List[Path] = []
		env_home = os.getenv("LIFTED_LINEAR_HOME")
		if env_home:
			roots.append(Path(env_home).expanduser())
		roots.extend(
			[
				repo_root / ".external" / "lifted-linear-gpp",
				repo_root / ".external" / "lifted-linear",
			],
		)
		seen: List[Path] = []
		for root in roots:
			resolved = root.resolve()
			if resolved not in seen and resolved.exists():
				seen.append(resolved)
		return tuple(seen)

	def _build_linearizer(self, root: Path) -> bool:
		linearizer_dir = root / "linearizer"
		if not (linearizer_dir / "Makefile").exists():
			return False
		env = os.environ.copy()
		path_entries: List[str] = []
		brew_bison = Path("/opt/homebrew/opt/bison/bin")
		if brew_bison.exists():
			path_entries.append(str(brew_bison))
		brew_bin = Path("/opt/homebrew/bin")
		if brew_bin.exists():
			path_entries.append(str(brew_bin))
		if path_entries:
			env["PATH"] = ":".join(path_entries + [env.get("PATH", "")])
		command = ["make", "-C", str(linearizer_dir)]
		if shutil.which("g++-14", path=env.get("PATH")):
			command.append("CXX=g++-14")
		result = subprocess.run(
			command,
			cwd=root,
			text=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
		return result.returncode == 0

	def _subprocess_env(self) -> Dict[str, str]:
		env = os.environ.copy()
		library_dirs = [str(path) for path in self._runtime_library_dirs()]
		if not library_dirs:
			return env
		for env_key in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
			current_entries = [
				entry
				for entry in str(env.get(env_key, "")).split(":")
				if entry
			]
			merged_entries: List[str] = []
			for entry in (*library_dirs, *current_entries):
				if entry not in merged_entries:
					merged_entries.append(entry)
			env[env_key] = ":".join(merged_entries)
		return env

	def _runtime_library_dirs(self) -> Tuple[Path, ...]:
		directories: List[Path] = []
		env_dir = os.getenv("CMS5_LIBDIR")
		if env_dir:
			directories.append(Path(env_dir).expanduser())
		directories.extend(
			[
				Path.home() / ".local" / "cryptominisat-ipasir" / "lib",
				Path.home() / ".local" / "src" / "cryptominisat" / "build" / "lib",
			],
		)
		unique: List[Path] = []
		for directory in directories:
			resolved = directory.expanduser().resolve()
			if resolved.exists() and resolved not in unique:
				unique.append(resolved)
		return tuple(unique)

	def _run_subprocess(
		self,
		command: Sequence[str],
		work_dir: Path,
		*,
		timeout_seconds: Optional[float],
		output_label: str,
	) -> Dict[str, Any]:
		return run_subprocess_to_files(
			command,
			work_dir=work_dir,
			output_label=output_label,
			timeout_seconds=timeout_seconds,
			env=self._subprocess_env(),
		)
