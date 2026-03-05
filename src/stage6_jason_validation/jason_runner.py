"""
Stage 6 Jason runner.

Runs generated AgentSpeak code with Jason (RunLocalMAS), boots a real Jason
`Environment` implementation for domain action semantics, and returns structured
validation metadata for pipeline logging.
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
from stage6_jason_validation.environment_adapter import (
	EnvironmentAdapterResult,
	Stage6EnvironmentAdapter,
	build_environment_adapter,
)


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
	javac_path: Optional[str]
	jason_jar: Optional[str]
	exit_code: Optional[int]
	timed_out: bool
	stdout: str
	stderr: str
	environment_adapter: Dict[str, Any]
	artifacts: Dict[str, str]

	def to_dict(self) -> Dict[str, Any]:
		return {
			"status": self.status,
			"backend": self.backend,
			"java_path": self.java_path,
			"java_version": self.java_version,
			"javac_path": self.javac_path,
			"jason_jar": self.jason_jar,
			"exit_code": self.exit_code,
			"timed_out": self.timed_out,
			"stdout": self.stdout,
			"stderr": self.stderr,
			"environment_adapter": dict(self.environment_adapter),
			"artifacts": dict(self.artifacts),
		}


class JasonRunner:
	"""Run Stage 5 AgentSpeak output in Jason and validate runtime outcomes."""

	backend_name = "RunLocalMAS"
	success_marker = "stage6 exec success"
	failure_marker = "stage6 exec failed"
	min_java_major = 17
	max_java_major = 23
	environment_class_name = "Stage6PipelineEnvironment"

	def __init__(
		self,
		*,
		stage6_dir: str | Path | None = None,
		timeout_seconds: int = 120,
		environment_adapter: Stage6EnvironmentAdapter | None = None,
		environment_adapter_name: str | None = None,
	) -> None:
		base_dir = (
			Path(stage6_dir).resolve()
			if stage6_dir is not None
			else Path(__file__).resolve().parent
		)
		self.stage6_dir = base_dir
		self.jason_src_dir = self.stage6_dir / "jason_src"
		self.timeout_seconds = timeout_seconds
		adapter_name = environment_adapter_name or os.getenv("STAGE6_ENV_ADAPTER")
		self.environment_adapter = environment_adapter or build_environment_adapter(adapter_name)

	def validate(
		self,
		*,
		agentspeak_code: str,
		target_literals: Sequence[HTNLiteral],
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str] = (),
		domain_name: str,
		output_dir: str | Path,
	) -> JasonValidationResult:
		"""Execute Jason validation and return a structured result."""

		if not action_schemas:
			raise JasonValidationError(
				"Stage 6 requires action schemas for real environment execution.",
				metadata={"action_schema_count": 0},
			)

		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)

		java_bin, java_major = self._select_java_binary()
		javac_bin = self._select_javac_binary(java_bin)
		jason_jar = self._ensure_jason_jar(java_bin)
		log_conf = self._resolve_log_config()

		runner_asl_path = output_path / "jason_runner_agent.asl"
		runner_mas2j_path = output_path / "jason_runner.mas2j"
		env_java_path = output_path / f"{self.environment_class_name}.java"
		env_class_path = output_path / f"{self.environment_class_name}.class"
		stdout_path = output_path / "jason_stdout.txt"
		stderr_path = output_path / "jason_stderr.txt"
		validation_json_path = output_path / "jason_validation.json"

		runner_asl = self._build_runner_asl(
			agentspeak_code,
			target_literals,
		)
		runner_mas2j = self._build_runner_mas2j(domain_name)
		env_source = self._build_environment_java_source(
			action_schemas=action_schemas,
			seed_facts=seed_facts,
			target_literals=target_literals,
		)
		runner_asl_path.write_text(runner_asl)
		runner_mas2j_path.write_text(runner_mas2j)
		env_java_path.write_text(env_source)
		self._compile_environment_java(
			java_bin=java_bin,
			javac_bin=javac_bin,
			jason_jar=jason_jar,
			env_java_path=env_java_path,
			output_path=output_path,
		)
		if not env_class_path.exists():
			raise JasonValidationError(
				"Stage 6 environment class compilation completed but class file is missing.",
				metadata={
					"environment_java": str(env_java_path),
					"environment_class": str(env_class_path),
				},
			)

		runtime_classpath = os.pathsep.join([str(jason_jar), str(output_path)])
		command = [
			java_bin,
			"-cp",
			runtime_classpath,
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
			"stage6_environment_java": str(env_java_path),
			"stage6_environment_class": str(env_class_path),
			"jason_stdout": str(stdout_path),
			"jason_stderr": str(stderr_path),
			"jason_validation": str(validation_json_path),
		}
		environment_result = self.environment_adapter.validate(stdout=stdout, stderr=stderr)
		is_success = self._is_successful_run(
			stdout=stdout,
			exit_code=exit_code,
			timed_out=timed_out,
			environment_result=environment_result,
		)
		status = "success" if is_success else "failed"
		result_payload = JasonValidationResult(
			status=status,
			backend=self.backend_name,
			java_path=java_bin,
			java_version=java_major,
			javac_path=javac_bin,
			jason_jar=str(jason_jar),
			exit_code=exit_code,
			timed_out=timed_out,
			stdout=stdout,
			stderr=stderr,
			environment_adapter=environment_result.to_dict(),
			artifacts=artifacts,
		)
		validation_json_path.write_text(json.dumps(result_payload.to_dict(), indent=2))

		if not is_success:
			failure_reason = self._failure_reason(
				stdout,
				stderr,
				exit_code,
				timed_out,
				environment_result,
			)
			raise JasonValidationError(
				f"Stage 6 Jason validation failed: {failure_reason}",
				metadata=result_payload.to_dict(),
			)

		return result_payload

	def toolchain_available(self) -> bool:
		"""Return whether Java+Jason requirements are available for Stage 6."""

		try:
			java_bin, _ = self._select_java_binary()
			self._select_javac_binary(java_bin)
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
		environment_ready_code = self._rewrite_primitive_wrappers_for_environment(agentspeak_code)
		target_context = self._target_context_expression(target_literals)
		lines = [
			environment_ready_code.rstrip(),
			"",
			"/* Stage 6 Execution Wrapper */",
			"!stage6_exec.",
			"",
		]
		lines.append(f"+!stage6_verify_targets : {target_context} <-")
		lines.extend(self._indent_body(["true"]))
		lines.append("")
		lines.append("+!stage6_exec : true <-")
		body_lines: List[str] = [
			'.print("stage6 exec start")',
			"!run_dfa",
			"?dfa_state(FINAL_STATE)",
			"?accepting_state(FINAL_STATE)",
			"!stage6_verify_targets",
			'.print("stage6 exec success")',
			".stopMAS",
		]
		lines.extend(self._indent_body(body_lines))
		lines.append("")
		lines.append("-!stage6_exec : true <-")
		lines.extend(self._indent_body(['.print("stage6 exec failed")', ".stopMAS"]))
		lines.append("")
		return "\n".join(lines)

	def _rewrite_primitive_wrappers_for_environment(self, agentspeak_code: str) -> str:
		start_marker = "/* Primitive Action Plans */"
		end_marker = "/* HTN Method Plans */"
		start_index = agentspeak_code.find(start_marker)
		end_index = agentspeak_code.find(end_marker)
		if start_index == -1 or end_index == -1 or end_index <= start_index:
			return agentspeak_code

		prefix = agentspeak_code[:start_index]
		section = agentspeak_code[start_index:end_index]
		suffix = agentspeak_code[end_index:]
		section_lines = section.splitlines()
		if not section_lines:
			return agentspeak_code

		header = section_lines[0]
		content_lines = section_lines[1:]
		chunks: List[List[str]] = []
		current: List[str] = []
		for line in content_lines:
			if not line.strip():
				if current:
					chunks.append(current)
					current = []
				continue
			current.append(line)
		if current:
			chunks.append(current)

		rewritten_chunks: List[str] = []
		for chunk in chunks:
			head_line = chunk[0]
			body_lines = chunk[1:]
			if not head_line.strip().startswith("+!"):
				rewritten_chunks.append("\n".join(chunk))
				continue
			if not body_lines:
				rewritten_chunks.append("\n".join(chunk))
				continue
			first_statement = body_lines[0].strip().rstrip(";.")
			if not first_statement:
				rewritten_chunks.append("\n".join(chunk))
				continue
			rewritten_chunks.append("\n".join([head_line, f"\t{first_statement}."]))

		rewritten_section = "\n\n".join([header, *rewritten_chunks]).rstrip() + "\n\n"
		return f"{prefix}{rewritten_section}{suffix}"

	def _build_runner_mas2j(self, domain_name: str) -> str:
		sanitized_domain = re.sub(r"[^a-zA-Z0-9_]+", "_", domain_name).strip("_").lower()
		if not sanitized_domain:
			sanitized_domain = "stage6"
		return (
			f"MAS stage6_{sanitized_domain} {{\n"
			f"    environment: {self.environment_class_name}\n"
			"    agents: jason_runner_agent;\n"
			"    aslSourcePath: \".\";\n"
			"}\n"
		)

	def _build_environment_java_source(
		self,
		*,
		action_schemas: Sequence[Dict[str, Any]],
		seed_facts: Sequence[str],
		target_literals: Sequence[HTNLiteral],
	) -> str:
		strong_predicate_keys = self._strong_negation_predicate_keys(
			action_schemas=action_schemas,
			target_literals=target_literals,
		)
		seed_atoms = [
			atom
			for atom in (self._hddl_fact_to_atom(fact) for fact in seed_facts)
			if atom is not None
		]
		seed_strong_negatives = [
			atom
			for atom in (
				self._hddl_fact_to_negative_atom(fact, strong_predicate_keys)
				for fact in seed_facts
			)
			if atom is not None
		]

		action_blocks: List[str] = []
		for schema in action_schemas:
			functor = schema.get("functor")
			if not functor:
				continue
			parameters = [str(item) for item in (schema.get("parameters") or [])]
			preconditions = list(schema.get("preconditions") or [])
			effects = list(schema.get("effects") or [])
			action_blocks.append(
				"""
		register(new ActionSchema(
			{functor},
			new String[]{{{parameters}}},
			new Pattern[]{{{preconditions}}},
			new Pattern[]{{{effects}}}
		));
		""".strip().format(
					functor=self._java_quote(functor),
					parameters=", ".join(self._java_quote(item) for item in parameters),
					preconditions=", ".join(self._render_pattern_java(item) for item in preconditions),
					effects=", ".join(self._render_pattern_java(item) for item in effects),
				),
			)

		seed_lines = "\n".join(
			f"\t\tworld.add({self._java_quote(atom)});"
			for atom in seed_atoms
		)
		seed_strong_negative_lines = "\n".join(
			f"\t\tstrongNegatives.add({self._java_quote(atom)});"
			for atom in seed_strong_negatives
		)
		action_lines = "\n\t\t".join(action_blocks)
		if not action_lines:
			action_lines = "// no action schemas"

		return f"""
import jason.asSyntax.Literal;
import jason.asSyntax.Structure;
import jason.environment.Environment;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class {self.environment_class_name} extends Environment {{

	private static final class Pattern {{
		final String predicate;
		final boolean positive;
		final String negationMode;
		final String[] args;

		Pattern(String predicate, boolean positive, String negationMode, String[] args) {{
			this.predicate = predicate;
			this.positive = positive;
			this.negationMode = negationMode == null ? "naf" : negationMode;
			this.args = args;
		}}

		boolean isStrongNegation() {{
			return "strong".equals(negationMode);
		}}
	}}

	private static final class ActionSchema {{
		final String name;
		final String[] parameters;
		final Pattern[] preconditions;
		final Pattern[] effects;

		ActionSchema(String name, String[] parameters, Pattern[] preconditions, Pattern[] effects) {{
			this.name = name;
			this.parameters = parameters;
			this.preconditions = preconditions;
			this.effects = effects;
		}}
	}}

	private final Set<String> world = new LinkedHashSet<>();
	private final Set<String> strongNegatives = new LinkedHashSet<>();
	private final Map<String, ActionSchema> actions = new HashMap<>();

	@Override
	public synchronized void init(String[] args) {{
		super.init(args);
		seedInitialFacts();
		loadActions();
		syncPercepts();
		System.out.println("stage6 env ready");
	}}

	@Override
	public synchronized boolean executeAction(String agName, Structure action) {{
		ActionSchema schema = actions.get(action.getFunctor());
		if (schema == null) {{
			System.out.println("stage6 env unknown action " + action);
			return false;
		}}
		if (action.getArity() != schema.parameters.length) {{
			System.out.println("stage6 env action failed " + action + " reason=arity");
			return false;
		}}

		Map<String, String> bindings = new HashMap<>();
		for (int i = 0; i < schema.parameters.length; i++) {{
			String parameter = canonical(schema.parameters[i]);
			String value = canonical(action.getTerm(i).toString());
			bindings.put(parameter, value);
			if (parameter.startsWith("?")) {{
				bindings.put(parameter.substring(1), value);
			}}
		}}

		if (!checkPreconditions(schema.preconditions, bindings)) {{
			System.out.println("stage6 env action failed " + action + " reason=precondition");
			return false;
		}}

		applyEffects(schema.effects, bindings);
		syncPercepts();
		System.out.println("stage6 env action success " + action);
		return true;
	}}

	private void seedInitialFacts() {{
		world.clear();
		strongNegatives.clear();
{seed_lines if seed_lines else ""}
{seed_strong_negative_lines if seed_strong_negative_lines else ""}
		for (String atom : world) {{
			strongNegatives.remove(atom);
		}}
		for (String atom : strongNegatives) {{
			world.remove(atom);
		}}
	}}

	private void loadActions() {{
		actions.clear();
		{action_lines}
	}}

	private void register(ActionSchema schema) {{
		actions.put(schema.name, schema);
	}}

	private boolean checkPreconditions(Pattern[] preconditions, Map<String, String> bindings) {{
		for (Pattern pattern : preconditions) {{
			if ("=".equals(pattern.predicate) && pattern.args.length == 2) {{
				String left = resolveToken(pattern.args[0], bindings);
				String right = resolveToken(pattern.args[1], bindings);
				boolean equal = left.equals(right);
				if (pattern.positive != equal) {{
					return false;
				}}
				continue;
			}}

			String grounded = ground(pattern.predicate, pattern.args, bindings);
			boolean holds;
			if (pattern.positive) {{
				holds = world.contains(grounded);
			}} else if (pattern.isStrongNegation()) {{
				holds = strongNegatives.contains(grounded);
			}} else {{
				holds = !world.contains(grounded);
			}}
			if (!holds) {{
				return false;
			}}
		}}
		return true;
	}}

	private void applyEffects(Pattern[] effects, Map<String, String> bindings) {{
		for (Pattern pattern : effects) {{
			if ("=".equals(pattern.predicate)) {{
				continue;
			}}
			String grounded = ground(pattern.predicate, pattern.args, bindings);
			if (pattern.positive) {{
				world.add(grounded);
				if (pattern.isStrongNegation()) {{
					strongNegatives.remove(grounded);
				}}
			}} else {{
				world.remove(grounded);
				if (pattern.isStrongNegation()) {{
					strongNegatives.add(grounded);
				}}
			}}
		}}
	}}

	private String ground(String predicate, String[] args, Map<String, String> bindings) {{
		if (args.length == 0) {{
			return predicate;
		}}
		String[] groundedArgs = Arrays.stream(args)
			.map(arg -> resolveToken(arg, bindings))
			.toArray(String[]::new);
		return predicate + "(" + String.join(",", groundedArgs) + ")";
	}}

	private String resolveToken(String rawToken, Map<String, String> bindings) {{
		String token = canonical(rawToken);
		if (bindings.containsKey(token)) {{
			return bindings.get(token);
		}}
		if (token.startsWith("?")) {{
			String bare = token.substring(1);
			if (bindings.containsKey(bare)) {{
				return bindings.get(bare);
			}}
		}}
		return token;
	}}

	private String canonical(String token) {{
		String value = token == null ? "" : token.trim();
		if (value.length() >= 2) {{
			boolean quoted =
				(value.startsWith("\\\"") && value.endsWith("\\\""))
				|| (value.startsWith("'") && value.endsWith("'"));
			if (quoted) {{
				value = value.substring(1, value.length() - 1);
			}}
		}}
		return value;
	}}

	private void syncPercepts() {{
		clearPercepts();
		for (String atom : world) {{
			addPercept(Literal.parseLiteral(atom));
		}}
		for (String atom : strongNegatives) {{
			addPercept(Literal.parseLiteral("~" + atom));
		}}
		informAgsEnvironmentChanged();
	}}
}}
""".strip() + "\n"

	def _compile_environment_java(
		self,
		*,
		java_bin: str,
		javac_bin: str,
		jason_jar: Path,
		env_java_path: Path,
		output_path: Path,
	) -> None:
		java_home = str(Path(java_bin).resolve().parent.parent)
		env = dict(os.environ)
		env["JAVA_HOME"] = java_home
		env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"
		compile_cmd = [
			javac_bin,
			"-cp",
			str(jason_jar),
			env_java_path.name,
		]
		result = subprocess.run(
			compile_cmd,
			cwd=output_path,
			text=True,
			capture_output=True,
			check=False,
			env=env,
		)
		if result.returncode == 0:
			return

		raise JasonValidationError(
			"Stage 6 environment Java compilation failed.",
			metadata={
				"java_bin": java_bin,
				"javac_bin": javac_bin,
				"environment_java": str(env_java_path),
				"stdout": result.stdout,
				"stderr": result.stderr,
				"return_code": result.returncode,
			},
		)

	@staticmethod
	def _java_quote(value: str) -> str:
		escaped = value.replace("\\", "\\\\").replace('"', '\\"')
		return f'"{escaped}"'

	def _render_pattern_java(self, payload: Dict[str, Any]) -> str:
		predicate = str(payload.get("predicate", ""))
		args = [str(item) for item in (payload.get("args") or [])]
		is_positive = bool(payload.get("is_positive", True))
		negation_mode = str(payload.get("negation_mode", "naf"))
		args_expr = ", ".join(self._java_quote(item) for item in args)
		return (
			f"new Pattern({self._java_quote(predicate)}, "
			f"{str(is_positive).lower()}, {self._java_quote(negation_mode)}, "
			f"new String[]{{{args_expr}}})"
		)

	def _target_context_expression(self, target_literals: Sequence[HTNLiteral]) -> str:
		rendered_literals = [
			self._literal_to_context_expression(literal)
			for literal in target_literals
		]
		rendered_literals = [item for item in rendered_literals if item]
		if not rendered_literals:
			return "true"
		return " & ".join(rendered_literals)

	@staticmethod
	def _literal_to_context_expression(literal: HTNLiteral) -> str:
		if literal.is_equality and len(literal.args) == 2:
			operator = "==" if literal.is_positive else "\\=="
			return f"{literal.args[0]} {operator} {literal.args[1]}"
		atom = (
			f"{literal.predicate}({', '.join(literal.args)})"
			if literal.args
			else literal.predicate
		)
		if literal.is_positive:
			return atom
		if literal.negation_mode == "strong":
			return f"~{atom}"
		return f"not {atom}"

	@staticmethod
	def _strong_negation_predicate_keys(
		*,
		action_schemas: Sequence[Dict[str, Any]],
		target_literals: Sequence[HTNLiteral],
	) -> set[str]:
		keys: set[str] = set()
		for literal in target_literals:
			if literal.is_positive or literal.is_equality:
				continue
			if literal.negation_mode != "strong":
				continue
			keys.add(f"{literal.predicate}/{len(literal.args)}")
		for schema in action_schemas:
			for group_name in ("preconditions", "effects"):
				for literal in schema.get(group_name, []) or []:
					predicate = str(literal.get("predicate", ""))
					if predicate == "=":
						continue
					arity = len(literal.get("args") or [])
					if literal.get("negation_mode") == "strong":
						keys.add(f"{predicate}/{arity}")
		return keys

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

	def _select_javac_binary(self, java_bin: str) -> str:
		java_home = Path(java_bin).resolve().parent.parent
		candidates = [
			str(java_home / "bin" / "javac"),
			shutil.which("javac") or "",
		]
		for candidate in candidates:
			if not candidate:
				continue
			path = Path(candidate)
			if path.exists() and os.access(path, os.X_OK):
				return str(path)
		raise JasonValidationError(
			"No javac binary found for Stage 6 environment compilation.",
			metadata={"java_bin": java_bin, "candidates": candidates},
		)

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
		environment_result: EnvironmentAdapterResult,
	) -> bool:
		if timed_out:
			return False
		if exit_code is None or exit_code != 0:
			return False
		if self.success_marker not in stdout:
			return False
		if self.failure_marker in stdout:
			return False
		if not environment_result.success:
			return False
		return True

	def _failure_reason(
		self,
		stdout: str,
		stderr: str,
		exit_code: Optional[int],
		timed_out: bool,
		environment_result: EnvironmentAdapterResult,
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
		if not environment_result.success:
			return (
				"environment adapter validation failed: "
				+ (environment_result.error or "unknown adapter error")
			)
		return "unknown validation error"

	@staticmethod
	def _hddl_fact_to_atom(fact: str) -> Optional[str]:
		text = (fact or "").strip()
		if not text.startswith("(") or not text.endswith(")"):
			return None
		inner = text[1:-1].strip()
		if not inner or inner.startswith("not "):
			return None
		tokens = inner.split()
		if not tokens:
			return None
		predicate, args = tokens[0], tokens[1:]
		if predicate == "=":
			return None
		if not args:
			return predicate
		return f"{predicate}({','.join(args)})"

	@staticmethod
	def _hddl_fact_to_negative_atom(
		fact: str,
		strong_predicate_keys: set[str],
	) -> Optional[str]:
		text = (fact or "").strip()
		if not text.startswith("(") or not text.endswith(")"):
			return None
		inner = text[1:-1].strip()
		if not inner.startswith("not "):
			return None
		negated = inner[4:].strip()
		if not negated.startswith("(") or not negated.endswith(")"):
			return None
		neg_inner = negated[1:-1].strip()
		tokens = neg_inner.split()
		if not tokens:
			return None
		predicate, args = tokens[0], tokens[1:]
		if predicate == "=":
			return None
		key = f"{predicate}/{len(args)}"
		if key not in strong_predicate_keys:
			return None
		if not args:
			return predicate
		return f"{predicate}({','.join(args)})"

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
