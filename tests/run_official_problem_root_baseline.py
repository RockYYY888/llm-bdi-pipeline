from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTEST_BIN = PROJECT_ROOT / ".venv" / "bin" / "pytest"
GENERATED_ROOT = PROJECT_ROOT / "tests" / "generated"
RUNS_ROOT = GENERATED_ROOT / "official_problem_root_full"

DOMAIN_TESTS = {
	"blocksworld": "tests/test_pipeline.py::test_blocksworld_official_problem_root_baseline",
	"marsrover": "tests/test_pipeline.py::test_marsrover_official_problem_root_baseline",
	"satellite": "tests/test_pipeline.py::test_satellite_official_problem_root_baseline",
	"transport": "tests/test_pipeline.py::test_transport_official_problem_root_baseline",
}


@dataclass
class DomainRun:
	name: str
	node_id: str
	command: List[str]
	output_path: Path
	junit_path: Path
	process: subprocess.Popen[str]


def _timestamp() -> str:
	return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _build_env() -> Dict[str, str]:
	env = os.environ.copy()
	env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
	env.setdefault("PANDA_PI_BIN", str(Path.home() / ".local" / "pandaPI-full" / "bin"))
	env.setdefault("PANDA_PI_ENGINE_MODES", "sat,default,bdd")
	return env


def _start_domain_run(run_dir: Path, name: str, node_id: str, env: Dict[str, str]) -> DomainRun:
	output_path = run_dir / f"{name}.out"
	junit_path = run_dir / f"{name}.xml"
	command = [
		str(PYTEST_BIN),
		"-q",
		node_id,
		f"--junitxml={junit_path}",
	]
	output_handle = output_path.open("w")
	process = subprocess.Popen(
		command,
		cwd=PROJECT_ROOT,
		env=env,
		stdout=output_handle,
		stderr=subprocess.STDOUT,
		text=True,
	)
	return DomainRun(
		name=name,
		node_id=node_id,
		command=command,
		output_path=output_path,
		junit_path=junit_path,
		process=process,
	)


def _parse_junit(junit_path: Path) -> Dict[str, object]:
	if not junit_path.exists():
		return {
			"tests": 0,
			"failures": 0,
			"errors": 0,
			"skipped": 0,
			"passed": 0,
			"failed_cases": [],
		}
	root = ET.fromstring(junit_path.read_text())
	testsuite = root if root.tag == "testsuite" else root.find("testsuite")
	if testsuite is None:
		return {
			"tests": 0,
			"failures": 0,
			"errors": 0,
			"skipped": 0,
			"passed": 0,
			"failed_cases": [],
		}
	tests = int(testsuite.attrib.get("tests", "0"))
	failures = int(testsuite.attrib.get("failures", "0"))
	errors = int(testsuite.attrib.get("errors", "0"))
	skipped = int(testsuite.attrib.get("skipped", "0"))
	failed_cases: List[Dict[str, str]] = []
	for testcase in testsuite.findall("testcase"):
		failure = testcase.find("failure")
		error = testcase.find("error")
		if failure is None and error is None:
			continue
		issue = failure if failure is not None else error
		failed_cases.append(
			{
				"name": testcase.attrib.get("name", ""),
				"classname": testcase.attrib.get("classname", ""),
				"message": issue.attrib.get("message", ""),
			},
		)
	return {
		"tests": tests,
		"failures": failures,
		"errors": errors,
		"skipped": skipped,
		"passed": max(tests - failures - errors - skipped, 0),
		"failed_cases": failed_cases,
	}


def main() -> int:
	run_dir = RUNS_ROOT / _timestamp()
	run_dir.mkdir(parents=True, exist_ok=True)
	env = _build_env()

	runs: List[DomainRun] = []
	for name, node_id in DOMAIN_TESTS.items():
		runs.append(_start_domain_run(run_dir, name, node_id, env))

	launch_metadata = {
		run.name: {
			"pid": run.process.pid,
			"node_id": run.node_id,
			"command": run.command,
			"output_path": str(run.output_path),
			"junit_path": str(run.junit_path),
		}
		for run in runs
	}
	(run_dir / "launch.json").write_text(json.dumps(launch_metadata, indent=2))

	domain_results: Dict[str, Dict[str, object]] = {}
	total_tests = 0
	total_passed = 0
	total_failed = 0
	total_skipped = 0

	for run in runs:
		return_code = run.process.wait()
		junit_summary = _parse_junit(run.junit_path)
		failed_count = int(junit_summary["failures"]) + int(junit_summary["errors"])
		passed_count = int(junit_summary["passed"])
		skipped_count = int(junit_summary["skipped"])
		test_count = int(junit_summary["tests"])
		total_tests += test_count
		total_passed += passed_count
		total_failed += failed_count
		total_skipped += skipped_count
		domain_results[run.name] = {
			"returncode": return_code,
			"tests": test_count,
			"passed": passed_count,
			"failed": failed_count,
			"skipped": skipped_count,
			"failed_cases": junit_summary["failed_cases"],
			"output_path": str(run.output_path),
			"junit_path": str(run.junit_path),
		}

	summary = {
		"run_dir": str(run_dir),
		"total_tests": total_tests,
		"total_passed": total_passed,
		"total_failed": total_failed,
		"total_skipped": total_skipped,
		"all_passed": total_failed == 0,
		"domains": domain_results,
	}
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

	lines = [
		f"run_dir: {run_dir}",
		f"total_tests: {total_tests}",
		f"total_passed: {total_passed}",
		f"total_failed: {total_failed}",
		f"total_skipped: {total_skipped}",
	]
	for name, result in domain_results.items():
		lines.append(
			f"{name}: {result['passed']}/{result['tests']} passed, {result['failed']} failed, {result['skipped']} skipped",
		)
		for failed_case in result["failed_cases"]:
			lines.append(f"  FAIL {failed_case['name']}: {failed_case['message']}")
	(run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
	print("\n".join(lines))
	return 0 if total_failed == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
