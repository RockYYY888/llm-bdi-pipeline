from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mainline_source_tree_omits_legacy_stage_keys() -> None:
	patterns = [
		"stage1_interpretation",
		"stage2_dfa_generation",
		"stage4_panda_planning",
		"stage5_agentspeak_rendering",
		"stage6_jason_validation",
		"stage1_status",
		"stage2_status",
		"stage3_status",
		"stage4_status",
		"stage5_status",
		"stage6_status",
		"stage7_status",
	]
	for pattern in patterns:
		result = subprocess.run(
			[
				"rg",
				"-n",
				pattern,
				"src/pipeline",
				"src/planning",
				"src/domain_build",
				"src/query_execution",
				"src/verification",
			],
			cwd=PROJECT_ROOT,
			text=True,
			capture_output=True,
		)
		assert result.returncode in (1, 0)
		assert result.stdout.strip() == "", result.stdout


def test_repository_tree_omits_retired_runtime_paths() -> None:
	retired_paths = [
		PROJECT_ROOT / "src" / "ltl_bdi_pipeline.py",
		PROJECT_ROOT / "src" / "pipeline_artifacts.py",
		PROJECT_ROOT / "src" / "stage1_interpretation",
		PROJECT_ROOT / "src" / "stage2_dfa_generation",
		PROJECT_ROOT / "src" / "stage3_method_synthesis",
		PROJECT_ROOT / "src" / "stage4_panda_planning",
		PROJECT_ROOT / "src" / "stage5_agentspeak_rendering",
		PROJECT_ROOT / "src" / "stage6_jason_validation",
		PROJECT_ROOT / "src" / "stage4_domain_gate",
		PROJECT_ROOT / "src" / "artifacts",
		PROJECT_ROOT / "src" / "experimental",
		PROJECT_ROOT / "src" / "external",
		PROJECT_ROOT / "src" / "utils" / "ipc_plan_verifier.py",
		PROJECT_ROOT / "src" / "utils" / "pipeline_logger.py",
		PROJECT_ROOT / "src" / "utils" / "benchmark_query_manifest.py",
		PROJECT_ROOT / "src" / "utils" / "setup_mona_path.py",
		PROJECT_ROOT / "tests" / "stage1_interpretation",
		PROJECT_ROOT / "tests" / "stage2_dfa_generation",
		PROJECT_ROOT / "tests" / "stage3_method_synthesis",
		PROJECT_ROOT / "tests" / "stage4_panda_planning",
		PROJECT_ROOT / "tests" / "stage5_agentspeak_rendering",
		PROJECT_ROOT / "tests" / "stage6_jason_validation",
		PROJECT_ROOT / "tests" / "test_pipeline.py",
		PROJECT_ROOT / "tests" / "test_pipeline_units.py",
		PROJECT_ROOT / "tests" / "domain_build",
		PROJECT_ROOT / "tests" / "planning",
		PROJECT_ROOT / "tests" / "query_execution",
		PROJECT_ROOT / "tests" / "verification",
		PROJECT_ROOT / "plan.md",
		PROJECT_ROOT / "docs" / "onion_verification_status.md",
	]
	for retired_path in retired_paths:
		assert not retired_path.exists(), retired_path
