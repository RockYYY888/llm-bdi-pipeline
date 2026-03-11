from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from utils.ipc_plan_verifier import IPCPlanVerifier


def test_render_primitive_only_plan_uses_official_primitive_plan_format():
	plan_text = IPCPlanVerifier.render_primitive_only_plan(
		[
			"unstack(b2,b3)",
			"put-down(b2)",
			"stack(b1,b4)",
		],
	)

	assert plan_text == "\n".join(
		[
			"==>",
			"0 unstack b2 b3",
			"1 put-down b2",
			"2 stack b1 b4",
			"root",
		],
	)


def test_parse_verifier_summary_distinguishes_primitive_success_from_full_htn_success():
	output = "\n".join(
		[
			"\u001b[0;34mPrimitive plan only (only valid if there are no method effects) ...\u001b[0m",
			"Primitive plan alone executable: true",
			"Plan verification result: false",
		],
	)

	clean = IPCPlanVerifier.strip_ansi(output)

	assert IPCPlanVerifier._extract_bool(clean, "Primitive plan alone executable") is True
	assert IPCPlanVerifier._extract_bool(clean, "Plan verification result") is False
	assert IPCPlanVerifier._infer_goal_reached(clean) is True
