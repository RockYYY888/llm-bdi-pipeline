"""
Masked-domain preparation and generated-domain materialization helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

from method_library.synthesis.schema import HTNMethodLibrary
from planning.panda_sat import PANDAPlanner
from utils.hddl_parser import HDDLParser


def strip_methods_from_domain_text(domain_text: str) -> str:
	"""
	Remove all top-level HDDL method blocks while preserving the rest of the domain.
	"""
	text = str(domain_text or "")
	token = "(:method"
	spans: list[Tuple[int, int]] = []
	cursor = 0
	while True:
		start = text.find(token, cursor)
		if start == -1:
			break
		end = HDDLParser._find_matching_paren(text, start)
		spans.append((start, end + 1))
		cursor = end + 1

	if not spans:
		return text if text.endswith("\n") else f"{text}\n"

	fragments: list[str] = []
	previous_end = 0
	for start, end in spans:
		fragments.append(text[previous_end:start])
		previous_end = end
	fragments.append(text[previous_end:])
	masked = "".join(fragments)
	masked = re.sub(r"\n{3,}", "\n\n", masked)
	return masked if masked.endswith("\n") else f"{masked}\n"


def write_masked_domain_file(
	*,
	official_domain_file: str | Path,
	output_path: str | Path,
) -> Dict[str, Any]:
	"""
	Strip methods from an official domain file and persist the masked domain.
	"""
	official_path = Path(official_domain_file).expanduser().resolve()
	target_path = Path(output_path).expanduser().resolve()
	original_text = official_path.read_text()
	original_domain = HDDLParser.parse_domain(str(official_path))
	masked_text = strip_methods_from_domain_text(original_text)
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(masked_text)
	masked_domain = HDDLParser.parse_domain(str(target_path))
	return {
		"official_domain_file": str(official_path),
		"masked_domain_file": str(target_path),
		"original_method_count": len(original_domain.methods),
		"masked_method_count": len(masked_domain.methods),
		"original_domain": original_domain,
		"masked_domain": masked_domain,
		"masked_domain_text": masked_text,
	}


def render_generated_domain_text(
	*,
	masked_domain_text: str,
	domain: Any,
	method_library: HTNMethodLibrary,
) -> str:
	"""
	Inject generated methods into a masked domain skeleton.
	"""
	planner = PANDAPlanner()
	task_lookup = {
		task.name: task
		for task in method_library.compound_tasks
	}
	task_type_map = planner._infer_task_type_map(domain, method_library)
	predicate_types = planner._predicate_type_map(domain)
	action_types = planner._action_type_map(domain)
	action_name_map = planner._action_export_name_map(domain, method_library)
	rendered_methods: list[str] = []
	for method in method_library.methods:
		rendered_methods.extend(
			planner._render_method(
				method,
				task_lookup,
				action_name_map,
				domain.types,
				task_type_map,
				predicate_types,
				action_types,
				export_source_names=True,
			),
		)
		rendered_methods.append("")
	method_block = "\n".join(rendered_methods).strip()
	if not method_block:
		return masked_domain_text if masked_domain_text.endswith("\n") else f"{masked_domain_text}\n"

	text = str(masked_domain_text or "")
	action_index = text.find("(:action")
	if action_index != -1:
		insert_at = action_index
	else:
		insert_at = text.rfind(")")
		if insert_at == -1:
			raise ValueError("Masked domain text does not contain a closing parenthesis.")

	prefix = text[:insert_at].rstrip()
	suffix = text[insert_at:].lstrip("\n")
	generated = f"{prefix}\n\n{method_block}\n\n{suffix}"
	return generated if generated.endswith("\n") else f"{generated}\n"


def write_generated_domain_file(
	*,
	masked_domain_text: str,
	domain: Any,
	method_library: HTNMethodLibrary,
	output_path: str | Path,
) -> Dict[str, Any]:
	"""
	Render and persist a generated domain file that contains generated methods only.
	"""
	target_path = Path(output_path).expanduser().resolve()
	generated_text = render_generated_domain_text(
		masked_domain_text=masked_domain_text,
		domain=domain,
		method_library=method_library,
	)
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(generated_text)
	generated_domain = HDDLParser.parse_domain(str(target_path))
	return {
		"generated_domain_file": str(target_path),
		"generated_domain_text": generated_text,
		"generated_domain": generated_domain,
		"generated_method_count": len(generated_domain.methods),
	}
