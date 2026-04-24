"""
Execution logger for the semantic domain-complete pipeline.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


STEP_TITLES = {
	"goal_grounding": "GOAL GROUNDING",
	"method_synthesis": "METHOD SYNTHESIS",
	"domain_gate": "DOMAIN GATE",
	"agentspeak_rendering": "AGENTSPEAK RENDERING",
	"plan_solve": "PLAN SOLVE",
	"runtime_execution": "RUNTIME EXECUTION",
	"plan_verification": "OFFICIAL VERIFICATION",
}

INLINE_LOG_SECTION_LIMIT_BYTES = 12_000
INLINE_TEXT_FIELD_LIMIT_CHARS = 2_000
LLM_PAYLOAD_KEYS = frozenset({"prompt", "response"})
EXTERNAL_ARTIFACT_KEYS = frozenset(
	{
		"method_library",
		"plan_library",
		"method_synthesis_metadata",
		"plan_library_asl",
	},
)


@dataclass
class ExecutionRecord:
	"""Structured record for one pipeline execution."""

	timestamp: str
	natural_language: str
	success: bool
	status: Optional[str] = None
	step: Optional[str] = None
	mode: str = "plan_library_evaluation"
	run_origin: str = "src"
	logs_root: str = "artifacts/runs"
	domain_name: Optional[str] = None
	problem_name: Optional[str] = None
	domain_file: str = ""
	problem_file: Optional[str] = None
	output_dir: str = "output"
	execution_time_seconds: float = 0.0
	timings: Dict[str, Any] = field(default_factory=dict)
	ltlf_formula: Optional[str] = None
	ltlf_atom_count: Optional[int] = None
	ltlf_operator_counts: Dict[str, Any] = field(default_factory=dict)
	jason_failure_class: Optional[str] = None
	failed_goals: Tuple[str, ...] = ()
	verifier_missing_goal_facts: Tuple[str, ...] = ()
	failure_signature: Optional[Dict[str, Any]] = None
	goal_grounding: Optional[Dict[str, Any]] = None
	method_synthesis: Optional[Dict[str, Any]] = None
	domain_gate: Optional[Dict[str, Any]] = None
	agentspeak_rendering: Optional[Dict[str, Any]] = None
	plan_solve: Optional[Dict[str, Any]] = None
	runtime_execution: Optional[Dict[str, Any]] = None
	plan_verification: Optional[Dict[str, Any]] = None


class ExecutionLogger:
	"""Persist semantic execution JSON and text logs."""

	def __init__(self, logs_dir: str = "logs", run_origin: str = "src") -> None:
		self.logs_dir = Path(logs_dir)
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		self.run_origin = run_origin
		self.current_record: Optional[ExecutionRecord] = None
		self.start_time: Optional[datetime] = None
		self.current_log_dir: Optional[Path] = None

	def start_pipeline(
		self,
		natural_language: str,
		mode: str = "plan_library_evaluation",
		domain_file: str = "",
		problem_file: str | None = None,
		domain_name: str | None = None,
		problem_name: str | None = None,
		output_dir: str = "output",
		timestamp: str | None = None,
	) -> None:
		self.start_time = datetime.now()
		if timestamp is None:
			timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

		resolved_domain_name = self._slug_component(domain_name or Path(domain_file).stem)
		resolved_problem_name = self._slug_component(problem_name) if problem_name else None
		dir_parts = [timestamp, resolved_domain_name]
		if resolved_problem_name:
			dir_parts.append(resolved_problem_name)
		dir_name = "_".join(part for part in dir_parts if part)
		self.current_log_dir = self.logs_dir / dir_name
		self.current_log_dir.mkdir(parents=True, exist_ok=True)

		self.current_record = ExecutionRecord(
			timestamp=timestamp,
			natural_language=self._compact_text_field(natural_language),
			success=False,
			mode=mode,
			run_origin=self.run_origin,
			logs_root=str(self.logs_dir),
			domain_name=domain_name,
			problem_name=problem_name,
			domain_file=domain_file,
			problem_file=problem_file,
			output_dir=str(self.current_log_dir),
		)
		self._save_current_state()

	def record_step_timing(
		self,
		step_name: str,
		total_seconds: float,
		*,
		breakdown: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		if self.current_record is None:
			return
		self.current_record.timings[step_name] = {
			"total_seconds": float(total_seconds),
			"breakdown": dict(breakdown or {}),
			"metadata": dict(metadata or {}),
		}
		self._save_current_state()

	def record_failure_signature(
		self,
		signature: Optional[Dict[str, Any]],
	) -> None:
		if self.current_record is None:
			return
		normalized_signature = self._sanitise_paths(dict(signature or {})) if signature is not None else {}
		normalized_signature = self._compact_failure_signature(normalized_signature)
		self.current_record.failure_signature = normalized_signature or None
		self.current_record.ltlf_formula = normalized_signature.get("ltlf_formula")
		self.current_record.ltlf_atom_count = normalized_signature.get("ltlf_atom_count")
		self.current_record.ltlf_operator_counts = dict(
			normalized_signature.get("ltlf_operator_counts") or {},
		)
		self.current_record.jason_failure_class = normalized_signature.get("jason_failure_class")
		self.current_record.failed_goals = tuple(
			str(goal).strip()
			for goal in (normalized_signature.get("failed_goals") or ())
			if str(goal).strip()
		)
		self.current_record.verifier_missing_goal_facts = tuple(
			str(fact).strip()
			for fact in (normalized_signature.get("verifier_missing_goal_facts") or ())
			if str(fact).strip()
		)
		self._save_current_state()

	def log_goal_grounding_success(
		self,
		artifacts: Dict[str, Any],
		*,
		used_llm: bool,
		model: Optional[str],
		llm_prompt: Optional[Dict[str, str]],
		llm_response: Optional[str],
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"goal_grounding",
			status="success",
			artifacts=artifacts,
			metadata=metadata,
			llm=(
				{
					"used": used_llm,
					"model": model,
					"prompt": llm_prompt,
					"response": llm_response,
				}
				if used_llm or model or llm_prompt or llm_response
				else None
			),
		)

	def log_goal_grounding_error(
		self,
		error: str,
		*,
		model: Optional[str] = None,
		llm_prompt: Optional[Dict[str, str]] = None,
		llm_response: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"goal_grounding",
			status="failed",
			error=str(error),
			metadata=metadata,
			llm=(
				{
					"used": bool(model or llm_prompt or llm_response),
					"model": model,
					"prompt": llm_prompt,
					"response": llm_response,
				}
				if model or llm_prompt or llm_response
				else None
			),
		)

	def log_method_synthesis(
		self,
		method_library: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		model: Optional[str] = None,
		llm_prompt: Optional[Dict[str, str]] = None,
		llm_response: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"method_synthesis",
			status=status.lower(),
			error=error,
			artifacts=(
				{"method_library": method_library}
				if method_library is not None
				else None
			),
			metadata=metadata,
			llm=(
				{
					"used": bool(model or llm_prompt or llm_response),
					"model": model,
					"prompt": llm_prompt,
					"response": llm_response,
				}
				if model or llm_prompt or llm_response
				else None
			),
		)

	def log_domain_gate(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		backend: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"domain_gate",
			status=status.lower(),
			backend=backend,
			error=error,
			artifacts=artifacts,
			metadata=metadata,
		)

	def log_agentspeak_rendering(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"agentspeak_rendering",
			status=status.lower(),
			error=error,
			artifacts=artifacts,
			metadata=metadata,
		)

	def log_plan_solve(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		backend: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"plan_solve",
			status=status.lower(),
			backend=backend,
			error=error,
			artifacts=artifacts,
			metadata=metadata,
		)

	def log_runtime_execution(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		backend: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"runtime_execution",
			status=status.lower(),
			backend=backend,
			error=error,
			artifacts=artifacts,
			metadata=metadata,
		)

	def log_official_verification(
		self,
		artifacts: Optional[Dict[str, Any]],
		status: str,
		*,
		error: Optional[str] = None,
		backend: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
	) -> None:
		self._set_step_payload(
			"plan_verification",
			status=status.lower(),
			backend=backend,
			error=error,
			artifacts=artifacts,
			metadata=metadata,
		)

	def update_step_artifacts(self, step_name: str, artifacts: Dict[str, Any]) -> None:
		"""Merge lightweight artifact references into an existing step payload."""
		if self.current_record is None:
			return
		current_payload = getattr(self.current_record, step_name, None)
		payload: Dict[str, Any] = (
			dict(current_payload)
			if isinstance(current_payload, dict)
			else {"status": "success"}
		)
		current_artifacts = payload.get("artifacts")
		merged_artifacts = (
			dict(current_artifacts)
			if isinstance(current_artifacts, dict)
			else {}
		)
		merged_artifacts.update(artifacts)
		payload["artifacts"] = self._compact_artifacts_payload(
			step_name,
			self._sanitise_paths(merged_artifacts),
		)
		setattr(self.current_record, step_name, payload)
		self._save_current_state()

	def end_pipeline(self, *, success: bool) -> Path:
		if self.current_record is None or self.current_log_dir is None:
			raise RuntimeError("No execution is currently active.")
		if self.start_time is not None:
			self.current_record.execution_time_seconds = (
				datetime.now() - self.start_time
			).total_seconds()
		self.current_record.success = bool(success)
		self.current_record.status = "success" if success else "failed"
		self._save_current_state()
		self._write_human_log()
		return self.current_log_dir / "execution.txt"

	def _set_step_payload(
		self,
		step_name: str,
		*,
		status: str,
		backend: Optional[str] = None,
		error: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
		artifacts: Optional[Dict[str, Any]] = None,
		llm: Optional[Dict[str, Any]] = None,
	) -> None:
		if self.current_record is None:
			return
		payload: Dict[str, Any] = {"status": status}
		if backend:
			payload["backend"] = backend
		if error:
			payload["error"] = str(error)
		if metadata:
			payload["metadata"] = self._compact_large_section(
				step_name,
				"metadata",
				self._sanitise_paths(metadata),
			)
		if artifacts:
			payload["artifacts"] = self._compact_artifacts_payload(
				step_name,
				self._sanitise_paths(artifacts),
			)
		if llm:
			payload["llm"] = self._compact_llm_payload(step_name, llm)
		if status == "failed":
			self.current_record.status = "failed"
			self.current_record.step = step_name
		setattr(self.current_record, step_name, payload)
		self._save_current_state()

	def _save_current_state(self) -> None:
		if self.current_record is None or self.current_log_dir is None:
			return
		self.current_log_dir.mkdir(parents=True, exist_ok=True)
		execution_path = self.current_log_dir / "execution.json"
		execution_path.write_text(json.dumps(self._record_to_dict(), indent=2))

	def _record_to_dict(self) -> Dict[str, Any]:
		if self.current_record is None:
			return {}
		record = asdict(self.current_record)
		filtered_record = {
			key: value
			for key, value in record.items()
			if value not in (None, {}, [], "")
		}
		return self._json_safe(filtered_record)

	def _write_human_log(self) -> None:
		if self.current_record is None or self.current_log_dir is None:
			return
		record = self._record_to_dict()
		mode = str(record.get("mode") or "").strip()
		if mode == "plan_library_generation":
			header = "PLAN LIBRARY GENERATION"
		elif mode == "plan_library_evaluation":
			header = "PLAN LIBRARY EVALUATION"
		elif mode == "official_problem_root_execution":
			header = "OFFICIAL PROBLEM ROOT EXECUTION"
		else:
			header = "PIPELINE EXECUTION"
		lines = [
			header,
			"=" * 80,
			f"Mode: {mode or 'unknown'}",
			f"Success: {record.get('success')}",
			f"Domain: {record.get('domain_name') or 'N/A'}",
			f"Problem: {record.get('problem_name') or 'N/A'}",
			f"Execution seconds: {record.get('execution_time_seconds', 0.0):.3f}",
			"",
		]
		failure_signature = record.get("failure_signature")
		if isinstance(failure_signature, dict) and failure_signature:
			lines.extend(
				[
					"FAILURE SIGNATURE",
					"-" * 80,
					json.dumps(failure_signature, indent=2),
					"",
				],
			)
		for step_name in (
			"goal_grounding",
			"method_synthesis",
			"domain_gate",
			"agentspeak_rendering",
			"plan_solve",
			"runtime_execution",
			"plan_verification",
		):
			payload = record.get(step_name)
			if not isinstance(payload, dict):
				continue
			lines.extend(
				[
					STEP_TITLES[step_name],
					"-" * 80,
					f"Status: {str(payload.get('status', '')).upper()}",
				]
			)
			if payload.get("backend"):
				lines.append(f"Backend: {payload['backend']}")
			if payload.get("error"):
				lines.append(f"Error: {payload['error']}")
			if payload.get("metadata"):
				lines.append("Metadata:")
				lines.append(json.dumps(payload["metadata"], indent=2))
			if payload.get("artifacts"):
				lines.append("Artifacts:")
				lines.append(json.dumps(payload["artifacts"], indent=2))
			if payload.get("llm"):
				lines.append("LLM:")
				lines.append(json.dumps(payload["llm"], indent=2))
			lines.append("")
		(self.current_log_dir / "execution.txt").write_text("\n".join(lines).rstrip() + "\n")

	def _compact_llm_payload(self, step_name: str, llm: Dict[str, Any]) -> Dict[str, Any]:
		compact_payload: Dict[str, Any] = {}
		for key, value in llm.items():
			if key in LLM_PAYLOAD_KEYS and value is not None:
				reference = self._write_payload_file(
					step_name=step_name,
					payload_name=f"llm_{key}",
					value=value,
				)
				compact_payload[f"{key}_file"] = reference["file"]
				compact_payload[f"{key}_bytes"] = reference["bytes"]
				continue
			compact_payload[key] = self._json_safe(value)
		return compact_payload

	def _compact_artifacts_payload(
		self,
		step_name: str,
		artifacts: Dict[str, Any],
	) -> Dict[str, Any]:
		compact_payload: Dict[str, Any] = {}
		for key, value in artifacts.items():
			if key in EXTERNAL_ARTIFACT_KEYS and value is not None:
				reference = self._write_payload_file(
					step_name=step_name,
					payload_name=str(key),
					value=value,
				)
				compact_payload[f"{key}_file"] = reference["file"]
				compact_payload[f"{key}_bytes"] = reference["bytes"]
				continue
			compact_payload[key] = self._json_safe(value)
		return self._compact_large_section(step_name, "artifacts", compact_payload)

	def _compact_large_section(
		self,
		step_name: str,
		payload_name: str,
		value: Any,
	) -> Any:
		payload_size = len(
			json.dumps(self._json_safe(value), default=str).encode("utf-8"),
		)
		if payload_size <= INLINE_LOG_SECTION_LIMIT_BYTES:
			return self._json_safe(value)
		reference = self._write_payload_file(
			step_name=step_name,
			payload_name=payload_name,
			value=value,
		)
		return {
			"payload_file": reference["file"],
			"payload_bytes": reference["bytes"],
		}

	def _write_payload_file(
		self,
		*,
		step_name: str,
		payload_name: str,
		value: Any,
	) -> Dict[str, Any]:
		if self.current_log_dir is None:
			return {"file": "", "bytes": 0}
		payload_dir = self.current_log_dir / "payloads"
		payload_dir.mkdir(parents=True, exist_ok=True)
		slug = self._slug_component(f"{step_name}_{payload_name}")
		is_text_payload = isinstance(value, str)
		suffix = ".txt" if is_text_payload else ".json"
		payload_path = payload_dir / f"{slug}{suffix}"
		if is_text_payload:
			payload_text = value
		else:
			payload_text = json.dumps(self._json_safe(value), indent=2, default=str)
		payload_path.write_text(payload_text, encoding="utf-8")
		return {
			"file": str(payload_path.relative_to(self.current_log_dir)),
			"bytes": len(payload_text.encode("utf-8")),
		}

	def _sanitise_paths(self, value: Any) -> Any:
		if self.current_log_dir is None:
			return value
		if isinstance(value, dict):
			return {key: self._sanitise_paths(item) for key, item in value.items()}
		if isinstance(value, list):
			return [self._sanitise_paths(item) for item in value]
		if isinstance(value, tuple):
			return [self._sanitise_paths(item) for item in value]
		if not isinstance(value, str):
			return value
		candidate = Path(value)
		if not candidate.is_absolute():
			return value
		try:
			return str(candidate.resolve().relative_to(self.current_log_dir.resolve()))
		except Exception:
			return value

	def _json_safe(self, value: Any) -> Any:
		if isinstance(value, dict):
			return {
				self._json_safe(key): self._json_safe(item)
				for key, item in value.items()
			}
		if isinstance(value, list):
			return [self._json_safe(item) for item in value]
		if isinstance(value, tuple):
			return [self._json_safe(item) for item in value]
		if isinstance(value, bytes):
			return value.decode("utf-8", errors="replace")
		if isinstance(value, Path):
			return str(value)
		if isinstance(value, (str, int, float, bool)) or value is None:
			return value
		return str(value)

	def _compact_failure_signature(self, signature: Dict[str, Any]) -> Dict[str, Any]:
		compact = dict(signature)
		ltlf_formula = compact.get("ltlf_formula")
		if isinstance(ltlf_formula, str):
			compact["ltlf_formula"] = self._compact_text_field(ltlf_formula)
			if len(ltlf_formula) > INLINE_TEXT_FIELD_LIMIT_CHARS:
				compact["ltlf_formula_chars"] = len(ltlf_formula)
				compact["ltlf_formula_sha256"] = hashlib.sha256(
					ltlf_formula.encode("utf-8"),
				).hexdigest()
				compact["ltlf_formula_truncated"] = True
		return compact

	@staticmethod
	def _compact_text_field(value: str) -> str:
		text = str(value or "")
		if len(text) <= INLINE_TEXT_FIELD_LIMIT_CHARS:
			return text
		sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
		return (
			f"{text[:INLINE_TEXT_FIELD_LIMIT_CHARS]}"
			f"... [truncated, chars={len(text)}, sha256={sha256}]"
		)

	@staticmethod
	def _slug_component(value: str | None) -> str:
		raw = str(value or "").strip()
		if not raw:
			return "unknown"
		return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "unknown"


__all__ = ["ExecutionLogger", "ExecutionRecord"]
