from pathlib import Path
import sys

_src_dir = str(Path(__file__).resolve().parents[2] / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from utils.hddl_parser import HDDLParser
from domain_model.materialization import write_masked_domain_file
from method_library.synthesis.domain_prompts import build_domain_prompt_analysis_payload
from method_library.synthesis.method_family_taxonomy import (
	DIRECT_LEAF,
	HIERARCHICAL_ORCHESTRATION,
	ALREADY_SATISFIED_GUARD,
	RECURSIVE_REFINEMENT,
	SUPPORT_THEN_LEAF,
	classify_domain_methods,
	infer_blueprint_family_archetypes,
)
from method_library.synthesis.synthesizer import HTNMethodSynthesizer


DOMAIN_FILES = {
	"blocksworld": "src/domains/blocksworld/domain.hddl",
	"transport": "src/domains/transport/domain.hddl",
	"satellite": "src/domains/satellite/domain.hddl",
	"marsrover": "src/domains/marsrover/domain.hddl",
}


def test_official_transport_methods_match_structural_taxonomy() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["transport"])
	classifications = {
		classification.method_name: classification
		for classification in classify_domain_methods(domain)
	}

	assert classifications["m-load"].archetype == DIRECT_LEAF
	assert classifications["m-unload"].archetype == DIRECT_LEAF
	assert classifications["m-drive-to-via"].archetype == RECURSIVE_REFINEMENT
	assert classifications["m-i-am-there"].archetype == DIRECT_LEAF
	assert classifications["m-deliver"].archetype == HIERARCHICAL_ORCHESTRATION


def test_official_blocksworld_methods_match_structural_taxonomy() -> None:
	domain = HDDLParser.parse_domain(DOMAIN_FILES["blocksworld"])
	classifications = {
		classification.method_name: classification
		for classification in classify_domain_methods(domain)
	}

	assert classifications["m0_do_put_on"].archetype == DIRECT_LEAF
	assert classifications["m4_do_move"].archetype == SUPPORT_THEN_LEAF
	assert classifications["m7_do_clear"].archetype == RECURSIVE_REFINEMENT


def test_official_satellite_and_marsrover_methods_match_structural_taxonomy() -> None:
	satellite = HDDLParser.parse_domain(DOMAIN_FILES["satellite"])
	satellite_classifications = {
		classification.method_name: classification
		for classification in classify_domain_methods(satellite)
	}
	marsrover = HDDLParser.parse_domain(DOMAIN_FILES["marsrover"])
	marsrover_classifications = {
		classification.method_name: classification
		for classification in classify_domain_methods(marsrover)
	}

	assert satellite_classifications["method3"].archetype == DIRECT_LEAF
	assert satellite_classifications["method4"].archetype == HIERARCHICAL_ORCHESTRATION
	assert marsrover_classifications["m-empty-store-1"].archetype == ALREADY_SATISFIED_GUARD
	assert marsrover_classifications["m-get_soil_data"].archetype == HIERARCHICAL_ORCHESTRATION
	assert marsrover_classifications["m-navigate_abs-4"].archetype == SUPPORT_THEN_LEAF


def test_transport_masked_blueprints_use_archetypes_instead_of_fake_headline_tasks(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["transport"],
		output_path=tmp_path / "masked_transport.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}

	assert tuple(blueprints["load"]["family_archetypes"]) == (DIRECT_LEAF,)
	assert tuple(blueprints["unload"]["family_archetypes"]) == (DIRECT_LEAF,)
	assert HIERARCHICAL_ORCHESTRATION in tuple(blueprints["deliver"]["family_archetypes"])
	assert RECURSIVE_REFINEMENT in tuple(blueprints["get-to"]["family_archetypes"])
	assert HIERARCHICAL_ORCHESTRATION not in tuple(blueprints["get-to"]["family_archetypes"])


def test_blocksworld_masked_blueprints_expose_method_family_archetypes(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["blocksworld"],
		output_path=tmp_path / "masked_blocksworld.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}

	assert ALREADY_SATISFIED_GUARD in tuple(blueprints["do_put_on"]["family_archetypes"])
	assert HIERARCHICAL_ORCHESTRATION in tuple(blueprints["do_put_on"]["family_archetypes"])
	assert SUPPORT_THEN_LEAF not in tuple(blueprints["do_put_on"]["family_archetypes"])
	assert SUPPORT_THEN_LEAF in tuple(blueprints["do_move"]["family_archetypes"])
	assert DIRECT_LEAF in tuple(blueprints["do_clear"]["family_archetypes"])


def test_marsrover_masked_blueprints_separate_supported_leaf_from_mission_tasks(tmp_path: Path) -> None:
	domain = write_masked_domain_file(
		official_domain_file=DOMAIN_FILES["marsrover"],
		output_path=tmp_path / "masked_marsrover.hddl",
	)["masked_domain"]
	analysis = HTNMethodSynthesizer()._analyse_domain_actions(domain)
	payload = build_domain_prompt_analysis_payload(domain, action_analysis=analysis)
	blueprints = {
		str(blueprint["task_name"]): blueprint
		for blueprint in payload["method_blueprints"]
	}

	assert tuple(blueprints["send_soil_data"]["family_archetypes"]) == (
			ALREADY_SATISFIED_GUARD,
		SUPPORT_THEN_LEAF,
	)
	assert HIERARCHICAL_ORCHESTRATION in tuple(blueprints["get_soil_data"]["family_archetypes"])


def test_blueprint_archetype_inference_handles_helper_leaf_and_recursive_cases() -> None:
	assert infer_blueprint_family_archetypes(
		{
			"headline_candidates": ["helper_only"],
			"direct_primitive_achievers": ["pick_up(?v, ?l, ?p)"],
			"headline_support_tasks": ["none"],
			"support_call_palette": ["none"],
			"support_task_hints": ["none"],
			"uncovered_prerequisite_families": ["none"],
			"method_family_schemas": ["none"],
			"witness_binding_required": True,
			"typed_task_signature": "load(?v:vehicle, ?l:location, ?p:package)",
		}
	) == (DIRECT_LEAF,)
	assert infer_blueprint_family_archetypes(
		{
			"headline_candidates": ["at"],
			"direct_primitive_achievers": ["none"],
			"headline_support_tasks": ["none"],
			"support_call_palette": ["none"],
			"support_task_hints": ["none"],
			"uncovered_prerequisite_families": ["none"],
			"method_family_schemas": [
				{"recursive_support_calls": ["get-to(?v, AUX_LOCATION1)"]},
			],
			"witness_binding_required": True,
			"typed_task_signature": "get-to(?v:vehicle, ?l:location)",
		}
	) == (ALREADY_SATISFIED_GUARD, RECURSIVE_REFINEMENT)
