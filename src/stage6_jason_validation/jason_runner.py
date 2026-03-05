"""
Stage 6 Jason runner.

Runs generated AgentSpeak code with Jason (RunLocalMAS) and returns
structured validation metadata for pipeline logging.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from stage3_method_synthesis.htn_schema import HTNLiteral


class JasonValidationError(RuntimeError):
    """Raised when Stage 6 Jason validation fails."""

    def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class JasonValidationResult:
    """Structured result for Stage 6 validation."""

    status: str
    backend: str
    java_path: Optional[str]
    java_version: Optional[int]
    jason_jar: Optional[str]
    exit_code: Optional[int]
    timed_out: bool
    stdout: str
    stderr: str
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "java_path": self.java_path,
            "java_version": self.java_version,
            "jason_jar": self.jason_jar,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": dict(self.artifacts),
        }


class JasonRunner:
    """Run Stage 5 AgentSpeak output in Jason and validate execution markers."""

    backend_name = "RunLocalMAS"
    success_marker = "stage6 exec success"
    failure_marker = "stage6 exec failed"
    min_java_major = 17
    max_java_major = 23

    def __init__(
        self,
        *,
        stage6_dir: str | Path | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        base_dir = (
            Path(stage6_dir).resolve()
            if stage6_dir is not None
            else Path(__file__).resolve().parent
        )
        self.stage6_dir = base_dir
        self.jason_src_dir = self.stage6_dir / "jason_src"
        self.timeout_seconds = timeout_seconds

    def validate(
        self,
        *,
        agentspeak_code: str,
        target_literals: Sequence[HTNLiteral],
        domain_name: str,
        output_dir: str | Path,
    ) -> JasonValidationResult:
        """Execute Jason validation and return a structured result."""

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        java_bin, java_major = self._select_java_binary()
        jason_jar = self._ensure_jason_jar(java_bin)
        log_conf = self._resolve_log_config()

        runner_asl_path = output_path / "jason_runner_agent.asl"
        runner_mas2j_path = output_path / "jason_runner.mas2j"
        stdout_path = output_path / "jason_stdout.txt"
        stderr_path = output_path / "jason_stderr.txt"
        validation_json_path = output_path / "jason_validation.json"

        runner_asl = self._build_runner_asl(agentspeak_code, target_literals)
        runner_mas2j = self._build_runner_mas2j(domain_name)
        runner_asl_path.write_text(runner_asl)
        runner_mas2j_path.write_text(runner_mas2j)

        command = [
            java_bin,
            "-cp",
            str(jason_jar),
            "jason.infra.local.RunLocalMAS",
            runner_mas2j_path.name,
            "--log-conf",
            str(log_conf),
        ]

        timed_out = False
        exit_code: Optional[int] = None
        raw_stdout = ""
        raw_stderr = ""

        try:
            result = subprocess.run(
                command,
                cwd=output_path,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
            exit_code = result.returncode
            raw_stdout = result.stdout
            raw_stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            raw_stdout = exc.stdout or ""
            raw_stderr = exc.stderr or ""

        stdout = self._combine_process_output(raw_stdout, raw_stderr)
        stderr = raw_stderr

        stdout_path.write_text(stdout)
        stderr_path.write_text(raw_stderr)

        artifacts = {
            "jason_runner_agent": str(runner_asl_path),
            "jason_runner_mas2j": str(runner_mas2j_path),
            "jason_stdout": str(stdout_path),
            "jason_stderr": str(stderr_path),
            "jason_validation": str(validation_json_path),
        }
        is_success = self._is_successful_run(
            stdout=stdout,
            exit_code=exit_code,
            timed_out=timed_out,
        )
        status = "success" if is_success else "failed"
        result_payload = JasonValidationResult(
            status=status,
            backend=self.backend_name,
            java_path=java_bin,
            java_version=java_major,
            jason_jar=str(jason_jar),
            exit_code=exit_code,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
        )
        validation_json_path.write_text(json.dumps(result_payload.to_dict(), indent=2))

        if not is_success:
            failure_reason = self._failure_reason(stdout, stderr, exit_code, timed_out)
            raise JasonValidationError(
                f"Stage 6 Jason validation failed: {failure_reason}",
                metadata=result_payload.to_dict(),
            )

        return result_payload

    def toolchain_available(self) -> bool:
        """Return whether Java+Jason requirements are available for Stage 6."""

        try:
            java_bin, _ = self._select_java_binary()
            self._ensure_jason_jar(java_bin)
            self._resolve_log_config()
            return True
        except Exception:
            return False

    def _build_runner_asl(
        self,
        agentspeak_code: str,
        target_literals: Sequence[HTNLiteral],
    ) -> str:
        seed_beliefs = self._seed_beliefs_from_literals(target_literals)
        lines = [agentspeak_code.rstrip(), "", "/* Stage 6 Execution Wrapper */", "!stage6_exec.", ""]
        lines.append("+!stage6_exec : true <-")

        body_lines: List[str] = [f"+{belief}" for belief in seed_beliefs]
        body_lines.extend(
            [
                '.print("stage6 exec start")',
                "!run_dfa",
                '.print("stage6 exec success")',
                ".stopMAS",
            ],
        )
        lines.extend(self._indent_body(body_lines))
        lines.append("")
        lines.append("-!stage6_exec : true <-")
        lines.extend(self._indent_body(['.print("stage6 exec failed")', ".stopMAS"]))
        lines.append("")
        return "\n".join(lines)

    def _build_runner_mas2j(self, domain_name: str) -> str:
        sanitized_domain = re.sub(r"[^a-zA-Z0-9_]+", "_", domain_name).strip("_").lower()
        if not sanitized_domain:
            sanitized_domain = "stage6"
        return (
            f"MAS stage6_{sanitized_domain} {{\n"
            "    infrastructure: Centralised\n"
            "    agents: jason_runner_agent;\n"
            "    aslSourcePath: \".\";\n"
            "}\n"
        )

    def _seed_beliefs_from_literals(self, target_literals: Sequence[HTNLiteral]) -> List[str]:
        beliefs: List[str] = []
        seen: set[str] = set()
        for literal in target_literals:
            if literal.is_equality or not literal.is_positive:
                continue
            belief = self._literal_to_atom(literal)
            if belief in seen:
                continue
            seen.add(belief)
            beliefs.append(belief)
        return beliefs

    @staticmethod
    def _literal_to_atom(literal: HTNLiteral) -> str:
        if literal.args:
            return f"{literal.predicate}({', '.join(literal.args)})"
        return literal.predicate

    def _resolve_log_config(self) -> Path:
        log_conf = (
            self.jason_src_dir
            / "jason-interpreter"
            / "src"
            / "main"
            / "resources"
            / "templates"
            / "console-info-logging.properties"
        )
        if log_conf.exists():
            return log_conf
        raise JasonValidationError(
            "Stage 6 log configuration file is missing.",
            metadata={"log_conf": str(log_conf)},
        )

    def _ensure_jason_jar(self, java_bin: str) -> Path:
        jar_path = self._find_jason_jar()
        if jar_path is not None:
            return jar_path

        self._build_jason_cli(java_bin)
        jar_path = self._find_jason_jar()
        if jar_path is not None:
            return jar_path

        raise JasonValidationError(
            "Jason CLI jar is unavailable after build.",
            metadata={
                "jason_src_dir": str(self.jason_src_dir),
            },
        )

    def _find_jason_jar(self) -> Optional[Path]:
        bin_dir = self.jason_src_dir / "jason-cli" / "build" / "bin"
        if not bin_dir.exists():
            return None

        jars = sorted(bin_dir.glob("jason-cli-all-*.jar"), key=self._jar_version_key, reverse=True)
        if not jars:
            return None
        return jars[0]

    @staticmethod
    def _jar_version_key(path: Path) -> Tuple[int, ...]:
        match = re.search(r"jason-cli-all-(\d+(?:\.\d+)*)\.jar$", path.name)
        if not match:
            return (0,)
        return tuple(int(item) for item in match.group(1).split("."))

    def _build_jason_cli(self, java_bin: str) -> None:
        gradlew = self.jason_src_dir / "gradlew"
        if not gradlew.exists():
            raise JasonValidationError(
                "Stage 6 Jason source directory is missing gradlew.",
                metadata={"gradlew": str(gradlew)},
            )

        java_home = str(Path(java_bin).resolve().parent.parent)
        env = dict(os.environ)
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"

        result = subprocess.run(
            [str(gradlew), "config"],
            cwd=self.jason_src_dir,
            text=True,
            capture_output=True,
            check=False,
            timeout=600,
            env=env,
        )
        if result.returncode == 0:
            return

        raise JasonValidationError(
            "Jason build failed while running './gradlew config'.",
            metadata={
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "java_home": java_home,
            },
        )

    def _select_java_binary(self) -> Tuple[str, int]:
        candidate_bins = self._discover_java_candidates()
        supported: List[Tuple[str, int]] = []
        unsupported: Dict[str, Optional[int]] = {}

        for candidate in candidate_bins:
            major = self._probe_java_binary(candidate)
            if major is None:
                unsupported[candidate] = None
                continue
            if self.min_java_major <= major <= self.max_java_major:
                supported.append((candidate, major))
            else:
                unsupported[candidate] = major

        if not supported:
            raise JasonValidationError(
                "No supported Java runtime found for Stage 6 (requires Java 17-23).",
                metadata={"candidates": unsupported},
            )

        supported.sort(key=lambda item: item[1], reverse=True)
        return supported[0]

    def _discover_java_candidates(self) -> List[str]:
        candidates: List[str] = []
        self._append_candidate(candidates, os.getenv("STAGE6_JAVA_BIN"))

        stage6_java_home = os.getenv("STAGE6_JAVA_HOME")
        if stage6_java_home:
            self._append_candidate(candidates, str(Path(stage6_java_home) / "bin" / "java"))

        java_home = os.getenv("JAVA_HOME")
        if java_home:
            self._append_candidate(candidates, str(Path(java_home) / "bin" / "java"))

        which_java = shutil.which("java")
        self._append_candidate(candidates, which_java)

        if os.name == "posix":
            for root in (
                Path.home() / "Library" / "Java" / "JavaVirtualMachines",
                Path("/Library/Java/JavaVirtualMachines"),
            ):
                if not root.exists():
                    continue
                for jdk_home in sorted(root.glob("*/Contents/Home/bin/java")):
                    self._append_candidate(candidates, str(jdk_home))

        return candidates

    @staticmethod
    def _append_candidate(candidates: List[str], candidate: Optional[str]) -> None:
        if not candidate:
            return
        resolved = str(Path(candidate).expanduser())
        if resolved in candidates:
            return
        candidates.append(resolved)

    @staticmethod
    def _probe_java_binary(java_bin: str) -> Optional[int]:
        java_path = Path(java_bin)
        if not java_path.exists():
            return None
        try:
            result = subprocess.run(
                [str(java_path), "-version"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except Exception:
            return None

        version_text = (result.stderr or "") + "\n" + (result.stdout or "")
        match = re.search(r'version "([^"]+)"', version_text)
        if not match:
            return None
        version = match.group(1)
        return JasonRunner._java_major_from_version(version)

    @staticmethod
    def _java_major_from_version(version: str) -> Optional[int]:
        if not version:
            return None
        parts = version.split(".")
        if parts[0] == "1" and len(parts) > 1:
            try:
                return int(parts[1])
            except ValueError:
                return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    def _is_successful_run(
        self,
        *,
        stdout: str,
        exit_code: Optional[int],
        timed_out: bool,
    ) -> bool:
        if timed_out:
            return False
        if exit_code is None or exit_code != 0:
            return False
        if self.success_marker not in stdout:
            return False
        if self.failure_marker in stdout:
            return False
        return True

    def _failure_reason(
        self,
        stdout: str,
        stderr: str,
        exit_code: Optional[int],
        timed_out: bool,
    ) -> str:
        if timed_out:
            return f"timeout ({self.timeout_seconds}s)"
        if exit_code is None:
            return "missing process exit code"
        if exit_code != 0:
            return f"process exited with code {exit_code}"
        if self.failure_marker in stdout:
            return "failure marker detected in stdout"
        if self.success_marker not in stdout:
            stderr_hint = stderr.strip().splitlines()[-1] if stderr.strip() else "none"
            return f"success marker missing (stderr tail: {stderr_hint})"
        return "unknown validation error"

    @staticmethod
    def _combine_process_output(stdout: str, stderr: str) -> str:
        if not stderr:
            return stdout
        if not stdout:
            return stderr
        separator = "" if stdout.endswith("\n") else "\n"
        return f"{stdout}{separator}{stderr}"

    @staticmethod
    def _indent_body(lines: Iterable[str]) -> List[str]:
        body_lines = list(lines)
        if not body_lines:
            return ["\ttrue."]
        rendered: List[str] = []
        last_index = len(body_lines) - 1
        for index, line in enumerate(body_lines):
            suffix = "." if index == last_index else ";"
            rendered.append(f"\t{line}{suffix}")
        return rendered
