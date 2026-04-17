#!/usr/bin/env python3
"""
Compatibility launcher for PANDA's translation-based Fast Downward backend.

PANDA PI still invokes Fast Downward with the historical `--internal-plan-file`
flag, while newer Fast Downward releases accept `--plan-file` instead. This
wrapper rewrites that argument and forwards execution to a locally available
Fast Downward installation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional


def _rewrite_arguments(arguments: Iterable[str]) -> List[str]:
	rewritten: List[str] = []
	iterator = iter(arguments)
	for token in iterator:
		if token == "--internal-plan-file":
			rewritten.append("--plan-file")
			rewritten.append(next(iterator, "sas_plan"))
			continue
		if token == "--evaluator":
			rewritten.append("--heuristic")
			rewritten.append(next(iterator, "h=blind()"))
			continue
		rewritten.append(token)
	return rewritten


def _candidate_paths(wrapper_path: Path) -> List[Path]:
	candidates: List[Optional[str | Path]] = [
		os.getenv("PANDA_PI_FAST_DOWNWARD_REAL"),
		os.getenv("FAST_DOWNWARD_REAL"),
		os.getenv("PANDA_PI_FAST_DOWNWARD"),
		os.getenv("FAST_DOWNWARD"),
		shutil.which("fast-downward.py"),
		shutil.which("downward"),
		shutil.which("fast-downward"),
	]
	downloads_dir = Path.home() / "Downloads"
	if downloads_dir.exists():
		candidates.extend(downloads_dir.rglob("fast-downward.py"))
	return [
		Path(candidate).expanduser().resolve()
		for candidate in candidates
		if candidate and Path(candidate).expanduser().resolve() != wrapper_path
	]


def _resolve_fast_downward(wrapper_path: Path) -> Optional[Path]:
	for candidate in _candidate_paths(wrapper_path):
		if candidate.exists():
			return candidate
	return None


def _extract_plan_file(arguments: List[str]) -> tuple[Optional[str], List[str]]:
	component_args: List[str] = []
	plan_file: Optional[str] = None
	index = 0
	while index < len(arguments):
		token = arguments[index]
		if token == "--plan-file":
			if index + 1 < len(arguments):
				plan_file = arguments[index + 1]
			index += 2
			continue
		component_args.append(token)
		index += 1
	return plan_file, component_args


def _has_explicit_input(arguments: List[str]) -> bool:
	for token in arguments:
		if token.startswith("-"):
			continue
		candidate = Path(token)
		if candidate.exists():
			return True
		if candidate.suffix.lower() in {".sas", ".pddl", ".hddl"}:
			return True
	return False


def main() -> int:
	wrapper_path = Path(__file__).resolve()
	target = _resolve_fast_downward(wrapper_path)
	if target is None:
		print("Fast Downward compatibility wrapper could not locate a real installation.", file=sys.stderr)
		return 2

	command = _rewrite_arguments(sys.argv[1:])
	plan_file, component_args = _extract_plan_file(command)
	temp_input_path: Optional[Path] = None
	build_name = os.getenv("FAST_DOWNWARD_BUILD", "release64").strip() or "release64"
	if not _has_explicit_input(component_args) and not sys.stdin.isatty():
		with tempfile.NamedTemporaryFile(
			mode="wb",
			suffix=".sas",
			delete=False,
		) as handle:
			handle.write(sys.stdin.buffer.read())
			temp_input_path = Path(handle.name)
		command = ["--build", build_name, "--search"]
		if plan_file:
			command.extend(["--plan-file", plan_file])
		command.append(str(temp_input_path))
		command.extend(component_args)
	else:
		command = ["--build", build_name, *command]
	if target.suffix == ".py":
		invocation = [sys.executable, str(target), *command]
	else:
		invocation = [str(target), *command]

	try:
		completed = subprocess.run(invocation, text=False)
		return int(completed.returncode)
	finally:
		if temp_input_path is not None:
			temp_input_path.unlink(missing_ok=True)


if __name__ == "__main__":
	raise SystemExit(main())
