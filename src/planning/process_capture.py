"""
Memory-safe subprocess capture helpers for planning backends.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT = 16384


def sanitize_process_output_label(label: str) -> str:
	"""Return a filesystem-safe stem for subprocess log files."""
	sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or "").strip())
	return sanitized.strip("._") or "process"


def terminate_process_group(process: subprocess.Popen[str]) -> None:
	"""Terminate a subprocess group, falling back to the direct child when needed."""
	try:
		os.killpg(os.getpgid(process.pid), signal.SIGKILL)
	except Exception:
		try:
			process.kill()
		except Exception:
			return


def read_process_output_preview(
	output_path: Path,
	*,
	preview_byte_limit: int = PROCESS_OUTPUT_PREVIEW_BYTE_LIMIT,
) -> Dict[str, Any]:
	"""
	Read a bounded preview of a subprocess output file.

	The preview keeps the start and end of oversized logs so failures remain debuggable
	without pulling the full file back into Python memory.
	"""
	if not output_path.exists():
		return {
			"text": "",
			"truncated": False,
			"byte_size": 0,
			"path": str(output_path),
		}

	byte_size = int(output_path.stat().st_size)
	if byte_size <= preview_byte_limit:
		return {
			"text": output_path.read_text(encoding="utf-8", errors="replace"),
			"truncated": False,
			"byte_size": byte_size,
			"path": str(output_path),
		}

	head_limit = max(preview_byte_limit // 2, 1)
	tail_limit = max(preview_byte_limit - head_limit, 1)
	with output_path.open("rb") as handle:
		head_bytes = handle.read(head_limit)
	with output_path.open("rb") as handle:
		handle.seek(max(byte_size - tail_limit, 0))
		tail_bytes = handle.read(tail_limit)
	truncated_bytes = max(byte_size - len(head_bytes) - len(tail_bytes), 0)
	preview_text = (
		head_bytes.decode("utf-8", errors="replace")
		+ f"\n...[truncated {truncated_bytes} bytes]...\n"
		+ tail_bytes.decode("utf-8", errors="replace")
	)
	return {
		"text": preview_text,
		"truncated": True,
		"byte_size": byte_size,
		"path": str(output_path),
	}


def read_full_process_output(output_path: str | Path) -> str:
	"""Read the full process output file only when the caller explicitly needs it."""
	path = Path(output_path)
	if not path.exists():
		return ""
	return path.read_text(encoding="utf-8", errors="replace")


def run_subprocess_to_files(
	command: Sequence[str],
	*,
	work_dir: Path,
	output_label: str,
	timeout_seconds: Optional[float] = None,
	env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
	"""Execute a subprocess while streaming stdout and stderr directly to log files."""
	label = sanitize_process_output_label(output_label)
	stdout_path = work_dir / f"{label}.stdout.log"
	stderr_path = work_dir / f"{label}.stderr.log"
	with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle:
		with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_handle:
			process = subprocess.Popen(
				command,
				cwd=work_dir,
				env=env,
				text=True,
				stdout=stdout_handle,
				stderr=stderr_handle,
				start_new_session=True,
			)
			timed_out = False
			try:
				process.wait(timeout=timeout_seconds)
			except subprocess.TimeoutExpired:
				timed_out = True
				terminate_process_group(process)
				process.wait()

	stdout_preview = read_process_output_preview(stdout_path)
	stderr_preview = read_process_output_preview(stderr_path)
	return {
		"returncode": 124 if timed_out else int(process.returncode or 0),
		"timed_out": timed_out,
		"stdout": stdout_preview["text"],
		"stderr": stderr_preview["text"],
		"stdout_path": stdout_preview["path"],
		"stderr_path": stderr_preview["path"],
		"stdout_truncated": bool(stdout_preview["truncated"]),
		"stderr_truncated": bool(stderr_preview["truncated"]),
		"stdout_byte_size": int(stdout_preview["byte_size"]),
		"stderr_byte_size": int(stderr_preview["byte_size"]),
	}
