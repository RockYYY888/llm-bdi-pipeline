from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
	sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from execution_logging.execution_logger import ExecutionLogger
from method_library.context import MethodLibrarySynthesisContext
from method_library.validation.validator import MethodLibraryValidator
from tests.support.plan_library_generation_support import (
	DOMAIN_FILES,
	GENERATED_DOMAIN_BUILDS_DIR,
	GENERATED_LOGS_DIR,
	build_official_method_library,
)


def run_official_domain_gate_preflight(domain_key: str) -> Dict[str, Any]:
	domain_file = DOMAIN_FILES[domain_key]
	context = MethodLibrarySynthesisContext(domain_file=domain_file)
	context.logger = ExecutionLogger(logs_dir=str(GENERATED_LOGS_DIR), run_origin="tests")
	context.logger.start_pipeline(
		f"Official domain gate preflight for {domain_key}",
		mode="official_domain_preflight",
		domain_file=domain_file,
		domain_name=context.domain.name,
		output_dir=str(GENERATED_LOGS_DIR),
	)
	context.output_dir = context.logger.current_log_dir
	if context.logger.current_record is not None and context.output_dir is not None:
		context.logger.current_record.output_dir = str(context.output_dir)
		context.logger._save_current_state()

	method_library = build_official_method_library(domain_file)
	domain_gate_summary = MethodLibraryValidator(context).validate(method_library)
	success = domain_gate_summary is not None
	log_path = context.logger.end_pipeline(success=success)
	log_dir = Path(log_path).parent
	execution = json.loads((log_dir / "execution.json").read_text())

	artifact_root = GENERATED_DOMAIN_BUILDS_DIR / "official_ground_truth" / domain_key
	artifact_root.mkdir(parents=True, exist_ok=True)
	artifact_path = artifact_root / "domain_gate.json"
	artifact_path.write_text(
		json.dumps(
			{
				"success": success,
				"domain_gate": domain_gate_summary,
				"log_dir": str(log_dir),
			},
			indent=2,
		)
	)
	return {
		"success": success,
		"domain_gate": domain_gate_summary,
		"log_dir": log_dir,
		"artifact_root": artifact_root,
		"execution": execution,
	}


__all__ = ["run_official_domain_gate_preflight"]
