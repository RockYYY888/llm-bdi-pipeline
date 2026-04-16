"""
Focused tests for the Stage 5 Jason-oriented AgentSpeak lowering pass.
"""

import sys
from pathlib import Path

_tests_dir = str(Path(__file__).parent.parent)
if _tests_dir not in sys.path:
	sys.path.insert(0, _tests_dir)
_src_dir = str(Path(__file__).parent.parent.parent / "src")
if _src_dir not in sys.path:
	sys.path.insert(0, _src_dir)

from stage5_agentspeak_rendering.asl_method_lowering import ASLMethodLowering
from stage3_method_synthesis.htn_schema import HTNMethodLibrary, HTNTask


def test_compile_method_plans_specialises_transport_like_helper_clause():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_in(PACKAGE, VEHICLE) : object_type(PACKAGE, package) & object_type(VEHICLE, vehicle) & object_type(LOCATION, location) & object_type(CAPACITY_NUMBER1, capacity_number) & object_type(CAPACITY_NUMBER2, capacity_number) & at(VEHICLE, LOCATION) & capacity_predecessor(CAPACITY_NUMBER1, CAPACITY_NUMBER2) <-
	.print("runtime trace method flat ", "m_helper_in_constructive_1", "|", PACKAGE, "|", VEHICLE);
	!helper_at(VEHICLE, LOCATION);
	!helper_capacity(VEHICLE, CAPACITY_NUMBER2);
	!pick_up(VEHICLE, LOCATION, PACKAGE, CAPACITY_NUMBER1, CAPACITY_NUMBER2).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(at truck-0 city-loc-2)",
			"(capacity-predecessor capacity-0 capacity-1)",
		],
		runtime_objects=(
			"package-0",
			"truck-0",
			"city-loc-2",
			"capacity-0",
			"capacity-1",
		),
		object_types={
			"package-0": "package",
			"truck-0": "truck",
			"city-loc-2": "location",
			"capacity-0": "capacity_number",
			"capacity-1": "capacity_number",
		},
		type_parent_map={
			"package": "locatable",
			"truck": "vehicle",
			"vehicle": "locatable",
			"locatable": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)

	assert '+!helper_in(PACKAGE, "truck-0") :' in rewritten
	assert '!helper_at("truck-0", "city-loc-2");' in rewritten
	assert '!helper_capacity("truck-0", "capacity-1");' in rewritten
	assert '!pick_up("truck-0", "city-loc-2", PACKAGE, "capacity-0", "capacity-1").' in rewritten


def test_compile_method_plans_adds_prefix_noop_context_before_generic_variant():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_in(ITEM, CARRIER) : in(ITEM, CARRIER) <-
	.print("runtime trace method flat ", "m_helper_in_noop", "|", ITEM, "|", CARRIER);
	true.

+!helper_capacity(CARRIER, LEVEL) : capacity(CARRIER, LEVEL) <-
	.print("runtime trace method flat ", "m_helper_capacity_noop", "|", CARRIER, "|", LEVEL);
	true.

+!deliver(ITEM, TARGET) : capacity_predecessor(LOW, HIGH) <-
	.print("runtime trace method flat ", "m_deliver", "|", ITEM, "|", TARGET);
	!helper_in(ITEM, CARRIER);
	!helper_capacity(CARRIER, LOW);
	!helper_at(CARRIER, TARGET);
	!drop(CARRIER, TARGET, ITEM, LOW, HIGH).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(in package-0 truck-0)", "(capacity truck-0 capacity-1)"],
		runtime_objects=("package-0", "truck-0", "capacity-1"),
		object_types={
			"package-0": "item",
			"truck-0": "carrier",
			"capacity-1": "level",
		},
		type_parent_map={
			"item": "object",
			"carrier": "object",
			"level": "object",
		},
	)

	deliver_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!deliver(")
	]

	assert deliver_chunks
	assert "in(" in deliver_chunks[0]
	assert "capacity(" in deliver_chunks[0]
	assert deliver_chunks[-1].startswith("+!deliver(ITEM, TARGET) : capacity_predecessor(LOW, HIGH)")


def test_compile_method_plans_preserves_domain_noop_order_for_runtime_progress_guards():
	lowering = ASLMethodLowering()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_on_table",
				parameters=("?x",),
				is_primitive=False,
				source_predicates=("ontable",),
			),
		],
	)
	agentspeak_code = """
/* HTN Method Plans */
+!do_on_table(BLOCK1) : clear(BLOCK1) & object_type(BLOCK1, block) <-
	.print("runtime trace method flat ", "m3_do_on_table", "|", BLOCK1);
	!nop.

+!do_on_table(BLOCK1) : clear(BLOCK1) & handempty & not ontable(BLOCK1) & on(BLOCK1, BLOCK2) & object_type(BLOCK1, block) & object_type(BLOCK2, block) <-
	.print("runtime trace method flat ", "m2_do_on_table", "|", BLOCK1);
	!unstack(BLOCK1, BLOCK2);
	!put_down(BLOCK1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(clear b26)", "(handempty)", "(on b26 b266)"],
		runtime_objects=("b26", "b266"),
		object_types={
			"b26": "block",
			"b266": "block",
		},
		type_parent_map={"block": "object"},
		method_library=method_library,
	)

	do_on_table_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!do_on_table(")
	]

	assert do_on_table_chunks
	assert '"m3_do_on_table"' in do_on_table_chunks[0]
	assert '"m2_do_on_table"' in do_on_table_chunks[-1]


def test_compile_method_plans_keeps_noop_first_when_context_supports_task_effect():
	lowering = ASLMethodLowering()
	method_library = HTNMethodLibrary(
		compound_tasks=[
			HTNTask(
				name="do_clear",
				parameters=("?x",),
				is_primitive=False,
				source_predicates=("clear",),
			),
		],
	)
	agentspeak_code = """
/* HTN Method Plans */
+!do_clear(BLOCK1) : not clear(BLOCK1) & on(BLOCK2, BLOCK1) & handempty & object_type(BLOCK1, block) & object_type(BLOCK2, block) <-
	.print("runtime trace method flat ", "m7_do_clear", "|", BLOCK1);
	!do_clear(BLOCK2);
	!do_on_table(BLOCK2).

+!do_clear(BLOCK1) : clear(BLOCK1) & object_type(BLOCK1, block) <-
	.print("runtime trace method flat ", "m6_do_clear", "|", BLOCK1);
	!nop.

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(clear b26)", "(handempty)", "(on b266 b26)"],
		runtime_objects=("b26", "b266"),
		object_types={
			"b26": "block",
			"b266": "block",
		},
		type_parent_map={"block": "object"},
		method_library=method_library,
	)

	do_clear_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!do_clear(")
	]

	assert do_clear_chunks
	assert '"m6_do_clear"' in do_clear_chunks[0]


def test_compile_method_plans_allows_single_noop_prefix_context_for_recursive_child():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, FROM) : at(ROVER, FROM) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", FROM);
	true.

+!helper_at(ROVER, TO) : can_traverse(ROVER, FROM, TO) <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TO);
	!helper_at(ROVER, FROM);
	!navigate(ROVER, FROM, TO).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(at rover0 waypoint2)", "(can-traverse rover0 waypoint2 waypoint3)"],
		runtime_objects=("rover0", "waypoint2", "waypoint3"),
		object_types={
			"rover0": "rover",
			"waypoint2": "waypoint",
			"waypoint3": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	assert any(
		chunk.startswith(
			"+!helper_at(ROVER, TO) : can_traverse(ROVER, FROM, TO) & at(ROVER, FROM)",
		)
		for chunk in rewritten.split("\n\n")
	)


def test_compile_method_plans_adds_reachable_source_guard_for_self_recursive_child():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint2)",
			"(can-traverse rover0 waypoint0 waypoint3)",
			"(can-traverse rover0 waypoint1 waypoint3)",
			"(can-traverse rover0 waypoint2 waypoint1)",
			"(visible waypoint0 waypoint3)",
			"(visible waypoint1 waypoint3)",
			"(visible waypoint2 waypoint1)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1", "waypoint2", "waypoint3"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint2": "waypoint",
			"waypoint3": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	assert any(
		chunk.startswith(
			"+!helper_at(rover0, waypoint3) : can_traverse(rover0, waypoint1, waypoint3)",
		)
		and "at(rover0, waypoint2)" in chunk.split("<-", 1)[0]
		and "!helper_at(rover0, waypoint1);" in chunk
		and "!navigate(rover0, waypoint1, waypoint3)." in chunk
		for chunk in rewritten.split("\n\n")
	)


def test_compile_method_plans_reuses_forward_reachable_sources_for_return_edges():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint0)",
			"(can-traverse rover0 waypoint0 waypoint1)",
			"(can-traverse rover0 waypoint1 waypoint0)",
			"(visible waypoint0 waypoint1)",
			"(visible waypoint1 waypoint0)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	assert '+!helper_at(rover0, waypoint0) : can_traverse(rover0, waypoint1, waypoint0)' in rewritten
	assert '& at(rover0, waypoint1)' in rewritten
	assert '!helper_at(rover0, waypoint1);' in rewritten
	assert '!navigate(rover0, waypoint1, waypoint0).' in rewritten


def test_compile_method_plans_orders_prefix_compound_bridge_before_shorter_handoff():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!do_observation(IMAGE_DIRECTION, MODE) : object_type(IMAGE_DIRECTION, image_direction) & object_type(MODE, mode) & object_type(INSTRUMENT, instrument) & object_type(SATELLITE, satellite) & on_board(INSTRUMENT, SATELLITE) & power_avail(SATELLITE) & calibration_target(INSTRUMENT, CALIB_DIRECTION) & pointing(SATELLITE, CALIB_DIRECTION) & pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT) <-
	.print("runtime trace method flat ", "method2", "|", IMAGE_DIRECTION, "|", MODE);
	pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT);
	!activate_instrument(SATELLITE, INSTRUMENT);
	!take_image(SATELLITE, IMAGE_DIRECTION, INSTRUMENT, MODE).

+!do_observation(IMAGE_DIRECTION, MODE) : object_type(IMAGE_DIRECTION, image_direction) & object_type(MODE, mode) & object_type(DIRECTION, direction) & object_type(SATELLITE, satellite) & object_type(INSTRUMENT, instrument) & on_board(INSTRUMENT, SATELLITE) & power_avail(SATELLITE) & calibration_target(INSTRUMENT, CALIB_DIRECTION) & pointing(SATELLITE, CALIB_DIRECTION) & pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT) <-
	.print("runtime trace method flat ", "method0", "|", IMAGE_DIRECTION, "|", MODE);
	pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT);
	!activate_instrument(SATELLITE, INSTRUMENT);
	!turn_to(SATELLITE, IMAGE_DIRECTION, DIRECTION);
	!take_image(SATELLITE, IMAGE_DIRECTION, INSTRUMENT, MODE).

+!activate_instrument(SATELLITE, INSTRUMENT) : on_board(INSTRUMENT, SATELLITE) & power_avail(SATELLITE) & calibration_target(INSTRUMENT, CALIB_DIRECTION) & pointing(SATELLITE, CALIB_DIRECTION) <-
	.print("runtime trace method flat ", "method5", "|", SATELLITE, "|", INSTRUMENT);
	!switch_on(INSTRUMENT, SATELLITE);
	!auto_calibrate(SATELLITE, INSTRUMENT).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(on-board instrument1 satellite1)",
			"(power-avail satellite1)",
			"(calibration-target instrument1 star0)",
			"(pointing satellite1 phenomenon2)",
		],
		runtime_objects=(
			"satellite1",
			"instrument1",
			"star0",
			"star5",
			"phenomenon2",
			"image1",
		),
		object_types={
			"satellite1": "satellite",
			"instrument1": "instrument",
			"star0": "calib_direction",
			"star5": "image_direction",
			"phenomenon2": "direction",
			"image1": "mode",
		},
		type_parent_map={
			"satellite": "object",
			"instrument": "object",
			"calib_direction": "direction",
			"image_direction": "direction",
			"direction": "object",
			"mode": "object",
		},
	)

	do_observation_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!do_observation(")
	]

	assert do_observation_chunks
	assert '!turn_to(' in do_observation_chunks[0]
	assert '"method0"' in do_observation_chunks[0]


def test_compile_method_plans_binds_later_primitive_source_from_current_context():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* Primitive Action Plans */
+!turn_to(SATELLITE, IMAGE_DIRECTION, DIRECTION) : object_type(SATELLITE, satellite) & object_type(IMAGE_DIRECTION, image_direction) & object_type(DIRECTION, direction) & pointing(SATELLITE, DIRECTION) <-
	turn_to(SATELLITE, IMAGE_DIRECTION, DIRECTION).

/* HTN Method Plans */
+!do_observation(IMAGE_DIRECTION, MODE) : object_type(IMAGE_DIRECTION, image_direction) & object_type(MODE, mode) & object_type(DIRECTION, direction) & object_type(SATELLITE, satellite) & object_type(INSTRUMENT, instrument) & on_board(INSTRUMENT, SATELLITE) & power_avail(SATELLITE) & calibration_target(INSTRUMENT, CALIB_DIRECTION) & pointing(SATELLITE, CALIB_DIRECTION) & pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT) <-
	.print("runtime trace method flat ", "method0", "|", IMAGE_DIRECTION, "|", MODE);
	pipeline.no_ancestor_goal(activate_instrument, SATELLITE, INSTRUMENT);
	!activate_instrument(SATELLITE, INSTRUMENT);
	!turn_to(SATELLITE, IMAGE_DIRECTION, DIRECTION);
	!take_image(SATELLITE, IMAGE_DIRECTION, INSTRUMENT, MODE).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[],
		runtime_objects=("satellite1", "instrument1", "star0", "star5", "image1"),
		object_types={
			"satellite1": "satellite",
			"instrument1": "instrument",
			"star0": "direction",
			"star5": "image_direction",
			"image1": "mode",
		},
		type_parent_map={
			"satellite": "object",
			"instrument": "object",
			"direction": "object",
			"image_direction": "direction",
			"mode": "object",
		},
	)

	do_observation_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!do_observation(")
	]

	assert any(
		'!turn_to(SATELLITE, IMAGE_DIRECTION, CALIB_DIRECTION);' in chunk
		for chunk in do_observation_chunks
	)


def test_compile_method_plans_does_not_ground_self_recursive_intermediate_from_disconnected_road():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* Primitive Action Plans */
+!drive(VEHICLE, LOCATION2, LOCATION1) : at(VEHICLE, LOCATION2) & road(LOCATION2, LOCATION1) <-
	drive(VEHICLE, LOCATION2, LOCATION1).

/* HTN Method Plans */
+!get_to(VEHICLE, LOCATION1) : at(VEHICLE, LOCATION1) <-
	.print("runtime trace method flat ", "m-i-am-there", "|", VEHICLE, "|", LOCATION1);
	true.

+!get_to(VEHICLE, LOCATION1) : road(LOCATION2, LOCATION1) & object_type(VEHICLE, vehicle) & object_type(LOCATION1, location) & object_type(LOCATION2, location) & LOCATION2 \\== LOCATION1 <-
	.print("runtime trace method flat ", "m-drive-to-via", "|", VEHICLE, "|", LOCATION1);
	pipeline.no_ancestor_goal(get_to, VEHICLE, LOCATION2);
	!get_to(VEHICLE, LOCATION2);
	!drive(VEHICLE, LOCATION2, LOCATION1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(at truck-0 city-loc-3)",
			"(road city-loc-4 city-loc-2)",
			"(road city-loc-1 city-loc-2)",
		],
		runtime_objects=("truck-0", "city-loc-1", "city-loc-2", "city-loc-3", "city-loc-4"),
		object_types={
			"truck-0": "vehicle",
			"city-loc-1": "location",
			"city-loc-2": "location",
			"city-loc-3": "location",
			"city-loc-4": "location",
		},
		type_parent_map={
			"vehicle": "object",
			"location": "object",
		},
	)

	assert '+!get_to(VEHICLE, "city-loc-2") : road("city-loc-4", "city-loc-2")' not in rewritten
	assert '!get_to(VEHICLE, "city-loc-4");' not in rewritten
	assert '& at("truck-0", "city-loc-4") & at("truck-0", "city-loc-1")' not in rewritten


def test_compile_method_plans_does_not_ground_multi_step_noop_prefix_witnesses():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!get_to(VEHICLE, LOCATION) : at(VEHICLE, LOCATION) <-
	.print("runtime trace method flat ", "m-i-am-there", "|", VEHICLE, "|", LOCATION);
	true.

+!deliver(PACKAGE, DESTINATION) : object_type(PACKAGE, package) & object_type(DESTINATION, location) & object_type(PICKUP, location) & object_type(VEHICLE, vehicle) & pipeline.no_ancestor_goal(get_to, VEHICLE, PICKUP) & pipeline.no_ancestor_goal(load, VEHICLE, PICKUP, PACKAGE) & pipeline.no_ancestor_goal(get_to, VEHICLE, DESTINATION) & pipeline.no_ancestor_goal(unload, VEHICLE, DESTINATION, PACKAGE) <-
	.print("runtime trace method flat ", "m-deliver", "|", PACKAGE, "|", DESTINATION);
	pipeline.no_ancestor_goal(get_to, VEHICLE, PICKUP);
	!get_to(VEHICLE, PICKUP);
	pipeline.no_ancestor_goal(load, VEHICLE, PICKUP, PACKAGE);
	!load(VEHICLE, PICKUP, PACKAGE);
	pipeline.no_ancestor_goal(get_to, VEHICLE, DESTINATION);
	!get_to(VEHICLE, DESTINATION);
	pipeline.no_ancestor_goal(unload, VEHICLE, DESTINATION, PACKAGE);
	!unload(VEHICLE, DESTINATION, PACKAGE).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(at truck-0 city-loc-2)"],
		runtime_objects=("truck-0", "city-loc-2", "city-loc-0", "package-0"),
		object_types={
			"truck-0": "vehicle",
			"city-loc-2": "location",
			"city-loc-0": "location",
			"package-0": "package",
		},
		type_parent_map={
			"vehicle": "object",
			"location": "object",
			"package": "object",
		},
	)

	deliver_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!deliver(")
	]

	assert deliver_chunks
	assert '!load("truck-0", "city-loc-2", PACKAGE);' not in rewritten
	assert any('!load(VEHICLE, PICKUP, PACKAGE);' in chunk for chunk in deliver_chunks)


def test_compile_method_plans_guards_body_goal_bindings_after_runtime_failure():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!deliver(PACKAGE, DESTINATION) : object_type(PACKAGE, package) & object_type(DESTINATION, location) & object_type(VEHICLE, vehicle) <-
	.print("runtime trace method flat ", "m-deliver", "|", PACKAGE, "|", DESTINATION);
	!get_to(VEHICLE, DESTINATION);
	!unload(VEHICLE, DESTINATION, PACKAGE).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[],
		runtime_objects=("truck-0", "city-loc-0", "package-0"),
		object_types={
			"truck-0": "vehicle",
			"city-loc-0": "location",
			"package-0": "package",
		},
		type_parent_map={
			"vehicle": "object",
			"location": "object",
			"package": "object",
		},
	)

	assert "not blocked_runtime_goal(get_to, VEHICLE, DESTINATION)" in rewritten
	assert "not blocked_runtime_goal(unload, VEHICLE, DESTINATION, PACKAGE)" in rewritten


def test_compile_method_plans_adds_body_noop_context_variant_for_later_child():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!navigate_abs(ROVER, WAYPOINT) : at(ROVER, WAYPOINT) & object_type(ROVER, rover) & object_type(WAYPOINT, waypoint) <-
	.print("runtime trace method flat ", "m-navigate-noop", "|", ROVER, "|", WAYPOINT);
	true.

+!get_image_data(OBJECTIVE, MODE) : visible_from(OBJECTIVE, WAYPOINT) & object_type(OBJECTIVE, objective) & object_type(MODE, mode) & object_type(ROVER, rover) & object_type(WAYPOINT, waypoint) <-
	.print("runtime trace method flat ", "m-get-image", "|", OBJECTIVE, "|", MODE);
	!calibrate_abs(ROVER, CAMERA);
	!navigate_abs(ROVER, WAYPOINT);
	!take_image(ROVER, WAYPOINT, OBJECTIVE, CAMERA, MODE).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[],
		runtime_objects=("rover0", "waypoint0", "objective0", "high_res"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"objective0": "objective",
			"high_res": "mode",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
			"objective": "object",
			"mode": "object",
		},
	)

	assert (
		"+!get_image_data(OBJECTIVE, MODE) : visible_from(OBJECTIVE, WAYPOINT)"
		" & object_type(OBJECTIVE, objective) & object_type(MODE, mode)"
		" & object_type(ROVER, rover) & object_type(WAYPOINT, waypoint)"
		" & at(ROVER, WAYPOINT)"
	) in rewritten


def test_compile_method_plans_uses_complete_recursive_graph_before_variant_cap():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

/* DFA Transition Wrappers */
""".strip() + "\n"

	seed_facts = [
		"(available rover0)",
		"(at rover0 zsource)",
		"(can-traverse rover0 a0 goal)",
		"(can-traverse rover0 a0 filler0)",
		"(can-traverse rover0 a0 filler1)",
		"(can-traverse rover0 a0 filler2)",
		"(can-traverse rover0 a0 filler3)",
		"(can-traverse rover0 a0 filler4)",
		"(can-traverse rover0 a0 filler5)",
		"(can-traverse rover0 a0 filler6)",
		"(can-traverse rover0 m1 a0)",
		"(can-traverse rover0 zsource m1)",
		"(visible a0 goal)",
		"(visible a0 filler0)",
		"(visible a0 filler1)",
		"(visible a0 filler2)",
		"(visible a0 filler3)",
		"(visible a0 filler4)",
		"(visible a0 filler5)",
		"(visible a0 filler6)",
		"(visible m1 a0)",
		"(visible zsource m1)",
	]
	runtime_objects = (
		"rover0",
		"a0",
		"goal",
		"m1",
		"zsource",
		"filler0",
		"filler1",
		"filler2",
		"filler3",
		"filler4",
		"filler5",
		"filler6",
	)
	object_types = {"rover0": "rover"}
	object_types.update({obj: "waypoint" for obj in runtime_objects if obj != "rover0"})

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=seed_facts,
		runtime_objects=runtime_objects,
		object_types=object_types,
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
		max_candidates_per_clause=8,
	)

	assert (
		"+!helper_at(rover0, a0) : can_traverse(rover0, m1, a0)"
		" & visible(m1, a0) & available(rover0) & object_type(rover0, rover)"
		" & object_type(m1, waypoint) & object_type(a0, waypoint)"
		" & m1 \\== a0 & at(rover0, zsource)"
	) in rewritten


def test_compile_method_plans_bounds_recursive_source_guard_binding_enumeration(monkeypatch):
	seen_caps = []
	original = ASLMethodLowering._candidate_bindings_for_local_witnesses

	def _capturing_candidate_bindings(self, *args, max_candidates_per_clause, **kwargs):
		assert max_candidates_per_clause != -1
		seen_caps.append(int(max_candidates_per_clause))
		return original(
			self,
			*args,
			max_candidates_per_clause=max_candidates_per_clause,
			**kwargs,
		)

	monkeypatch.setattr(
		ASLMethodLowering,
		"_candidate_bindings_for_local_witnesses",
		_capturing_candidate_bindings,
	)

	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 zsource)",
			"(can-traverse rover0 m1 a0)",
			"(can-traverse rover0 zsource m1)",
			"(visible m1 a0)",
			"(visible zsource m1)",
		],
		runtime_objects=("rover0", "a0", "m1", "zsource"),
		object_types={
			"rover0": "rover",
			"a0": "waypoint",
			"m1": "waypoint",
			"zsource": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
		max_candidates_per_clause=8,
	)

	assert seen_caps
	assert max(seen_caps) <= 256
	assert "+!helper_at(rover0, a0)" in rewritten


def test_compile_method_plans_replaces_child_state_context_with_outer_source_guard():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET & at(ROVER, PREDECESSOR) <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint0)",
			"(can-traverse rover0 waypoint0 waypoint1)",
			"(can-traverse rover0 waypoint1 waypoint2)",
			"(can-traverse rover0 waypoint2 waypoint0)",
			"(visible waypoint0 waypoint1)",
			"(visible waypoint1 waypoint2)",
			"(visible waypoint2 waypoint0)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1", "waypoint2"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint2": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	assert '+!helper_at(rover0, waypoint0) : can_traverse(rover0, waypoint2, waypoint0)' in rewritten
	assert '& at(rover0, waypoint1)' in rewritten
	assert '!helper_at(rover0, waypoint2);' in rewritten
	assert 'at(rover0, waypoint2) & at(rover0, waypoint1)' not in rewritten


def test_compile_method_plans_discards_conflicting_recursive_body_goal_specialisations():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!get_soil_data(WAYPOINT) : store_of(STORE, ROVER) & equipped_for_soil_analysis(ROVER) & object_type(WAYPOINT, waypoint) & object_type(ROVER, rover) & object_type(STORE, store) & can_traverse(ROVER, MID, WAYPOINT) & can_traverse(ROVER, SOURCE, MID) & not can_traverse(ROVER, SOURCE, WAYPOINT) & not at(ROVER, WAYPOINT) & at(ROVER, WAYPOINT) & available(ROVER) & at(ROVER, SOURCE) & visible(SOURCE, MID) & visible(MID, WAYPOINT) & not visited(MID) <-
	.print("runtime trace method flat ", "m_get_soil_data_bad_recursive", "|", WAYPOINT);
	!navigate_abs(ROVER, WAYPOINT);
	!empty_store(STORE, ROVER);
	!sample_soil(ROVER, STORE, WAYPOINT);
	!send_soil_data(ROVER, WAYPOINT).

+!get_soil_data(WAYPOINT) : store_of(STORE, ROVER) & equipped_for_soil_analysis(ROVER) & object_type(WAYPOINT, waypoint) & object_type(ROVER, rover) & object_type(STORE, store) & can_traverse(ROVER, MID, WAYPOINT) & can_traverse(ROVER, SOURCE, MID) & not can_traverse(ROVER, SOURCE, WAYPOINT) & not at(ROVER, WAYPOINT) & available(ROVER) & at(ROVER, SOURCE) & visible(SOURCE, MID) & visible(MID, WAYPOINT) & not visited(MID) <-
	.print("runtime trace method flat ", "m_get_soil_data_good_recursive", "|", WAYPOINT);
	!navigate_abs(ROVER, WAYPOINT);
	!empty_store(STORE, ROVER);
	!sample_soil(ROVER, STORE, WAYPOINT);
	!send_soil_data(ROVER, WAYPOINT).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover1)",
			"(at rover1 waypoint1)",
			"(store-of rover1store rover1)",
			"(equipped-for-soil-analysis rover1)",
			"(can-traverse rover1 waypoint1 waypoint4)",
			"(can-traverse rover1 waypoint4 waypoint5)",
			"(can-traverse rover1 waypoint5 waypoint2)",
			"(visible waypoint1 waypoint4)",
			"(visible waypoint4 waypoint5)",
			"(visible waypoint5 waypoint2)",
		],
		runtime_objects=(
			"rover1",
			"rover1store",
			"waypoint1",
			"waypoint2",
			"waypoint4",
			"waypoint5",
		),
		object_types={
			"rover1": "rover",
			"rover1store": "store",
			"waypoint1": "waypoint",
			"waypoint2": "waypoint",
			"waypoint4": "waypoint",
			"waypoint5": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"store": "object",
			"waypoint": "object",
		},
	)

	get_soil_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!get_soil_data(")
	]

	assert get_soil_chunks
	assert any('"m_get_soil_data_good_recursive"' in chunk for chunk in get_soil_chunks)
	assert not any('"m_get_soil_data_bad_recursive"' in chunk for chunk in get_soil_chunks)
	assert not any(
		"not at(ROVER, WAYPOINT)" in chunk and "& at(ROVER, WAYPOINT) &" in chunk
		for chunk in get_soil_chunks
	)


def test_compile_method_plans_does_not_bind_empty_store_variable_from_navigation_witness():
	lowering = ASLMethodLowering()
	agentspeak_code = """
+!navigate_abs(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_navigate_abs_noop", "|", ROVER, "|", LOCATION);
	true.

+!navigate_abs(ROVER, TARGET) : can_traverse(ROVER, MID, TARGET) & visible(MID, TARGET) & not at(ROVER, TARGET) & can_traverse(ROVER, SOURCE, MID) & not can_traverse(ROVER, SOURCE, TARGET) & not visited(MID) & available(ROVER) & at(ROVER, SOURCE) & visible(SOURCE, MID) & object_type(ROVER, rover) & object_type(SOURCE, waypoint) & object_type(MID, waypoint) & object_type(TARGET, waypoint) <-
	.print("runtime trace method flat ", "m_navigate_abs_recursive", "|", ROVER, "|", TARGET);
	!navigate_abs(ROVER, MID);
	!navigate(ROVER, MID, TARGET).

+!empty_store(STORE, ROVER) : store_of(STORE, ROVER) & full(STORE) & object_type(STORE, store) & object_type(ROVER, rover) <-
	.print("runtime trace method flat ", "m_empty_store", "|", STORE, "|", ROVER);
	!drop(ROVER, STORE).

/* HTN Method Plans */
+!get_soil_data(WAYPOINT) : store_of(STORE, ROVER) & equipped_for_soil_analysis(ROVER) & object_type(WAYPOINT, waypoint) & object_type(ROVER, rover) & object_type(STORE, store) <-
	.print("runtime trace method flat ", "m_get_soil_data", "|", WAYPOINT);
	!navigate_abs(ROVER, WAYPOINT);
	!empty_store(STORE, ROVER);
	!sample_soil(ROVER, STORE, WAYPOINT);
	!send_soil_data(ROVER, WAYPOINT).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover1)",
			"(at rover1 waypoint1)",
			"(store-of rover1store rover1)",
			"(full rover1store)",
			"(equipped-for-soil-analysis rover1)",
			"(can-traverse rover1 waypoint1 waypoint4)",
			"(can-traverse rover1 waypoint4 waypoint5)",
			"(can-traverse rover1 waypoint5 waypoint2)",
			"(visible waypoint1 waypoint4)",
			"(visible waypoint4 waypoint5)",
			"(visible waypoint5 waypoint2)",
		],
		runtime_objects=(
			"rover1",
			"rover1store",
			"waypoint1",
			"waypoint2",
			"waypoint4",
			"waypoint5",
		),
		object_types={
			"rover1": "rover",
			"rover1store": "store",
			"waypoint1": "waypoint",
			"waypoint2": "waypoint",
			"waypoint4": "waypoint",
			"waypoint5": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"store": "object",
			"waypoint": "object",
		},
	)

	get_soil_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!get_soil_data(")
	]

	assert get_soil_chunks
	assert any("store_of(rover1store, rover1)" in chunk for chunk in get_soil_chunks)
	assert not any("object_type(WAYPOINT2, store)" in chunk for chunk in get_soil_chunks)
	assert not any("object_type(MID, store)" in chunk for chunk in get_soil_chunks)
	assert not any("object_type(SOURCE, store)" in chunk for chunk in get_soil_chunks)


def test_compile_method_plans_does_not_rebind_dfa_transition_trigger_args():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_clear(BLOCK) : clear(BLOCK) <-
	.print("runtime trace method flat ", "m_helper_clear_noop", "|", BLOCK);
	true.

+!dfa_step_q1_q2_on_b106_b97(BLOCK1, BLOCK2) : on(BLOCK3, BLOCK1) & object_type(BLOCK1, block) & object_type(BLOCK2, block) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q2_on_b106_b97_constructive_1", "|", BLOCK1, "|", BLOCK2);
	!helper_clear(BLOCK3);
	!stack(BLOCK1, BLOCK2).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(clear b113)"],
		runtime_objects=("b97", "b106", "b113"),
		object_types={
			"b97": "block",
			"b106": "block",
			"b113": "block",
		},
		type_parent_map={"block": "object"},
	)

	assert '+!dfa_step_q1_q2_on_b106_b97(BLOCK1, BLOCK2) :' in rewritten
	assert '+!dfa_step_q1_q2_on_b106_b97("b113", BLOCK2) :' not in rewritten
	assert '+!dfa_step_q1_q2_on_b106_b97("b113", "b97") :' not in rewritten


def test_goal_has_noop_runtime_support_uses_existence_check_without_materialising_bindings():
	class ExistenceOnlyLowering(ASLMethodLowering):
		def _noop_support_bindings(self, *args, **kwargs):
			raise AssertionError("ordering should not enumerate all noop support bindings")

	lowering = ExistenceOnlyLowering()

	assert lowering._goal_has_noop_runtime_support(
		goal_task_name="helper_at",
		goal_args=("rover0", "waypoint1"),
		noop_specs_by_task={
			"helper_at": [
				{
					"head_args": ("ROVER", "LOCATION"),
					"context_atoms": (
						{
							"predicate": "at",
							"args": ("ROVER", "LOCATION"),
						},
					),
					"inequalities": (),
				},
			],
		},
		fact_index={
			("at", 2): (
				("rover0", "waypoint0"),
				("rover0", "waypoint1"),
			),
		},
		type_domains={},
		caller_type_constraints=(),
		caller_inequalities=(),
	) is True


def test_noop_support_bindings_respects_max_bindings_cap():
	lowering = ASLMethodLowering()

	bindings = lowering._noop_support_bindings(
		head_args=("BLOCK",),
		goal_args=("b1",),
		context_atoms=(
			{
				"predicate": "on",
				"args": ("TOP", "BLOCK"),
			},
		),
		inequalities=(),
		fact_index={
			("on", 2): (
				("b2", "b1"),
				("b3", "b1"),
				("b4", "b1"),
			),
		},
		type_domains={},
		caller_type_constraints=(),
		caller_inequalities=(),
		max_bindings=2,
	)

	assert len(bindings) == 2
	assert all(binding.get("TOP") in {"b2", "b3", "b4"} for binding in bindings)


def test_compile_method_plans_prunes_large_local_witness_space_instead_of_dropping_specialisation():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!send_probe(OBJECTIVE, MODE) : object_type(OBJECTIVE, objective) & object_type(MODE, mode) & object_type(ROVER, rover) & object_type(CAMERA, camera) & object_type(IMAGE_WAYPOINT, waypoint) & object_type(COMMS_WAYPOINT, waypoint) & object_type(LANDER, lander) & object_type(LANDER_WAYPOINT, waypoint) & supports(CAMERA, MODE) & on_board(CAMERA, ROVER) & visible_from(OBJECTIVE, IMAGE_WAYPOINT) & at_lander(LANDER, LANDER_WAYPOINT) & visible(COMMS_WAYPOINT, LANDER_WAYPOINT) & channel_free(LANDER) <-
	.print("runtime trace method flat ", "m_send_probe_constructive_1", "|", OBJECTIVE, "|", MODE);
	!helper_have_image(ROVER, OBJECTIVE, MODE);
	!helper_at(ROVER, COMMS_WAYPOINT);
	!communicate_image_data(ROVER, LANDER, OBJECTIVE, MODE, COMMS_WAYPOINT, LANDER_WAYPOINT).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(supports camera0 high_res)",
			"(supports camera1 high_res)",
			"(supports camera2 high_res)",
			"(on-board camera0 rover0)",
			"(on-board camera1 rover1)",
			"(on-board camera2 rover2)",
			"(visible-from objective0 waypoint0)",
			"(visible-from objective0 waypoint1)",
			"(visible-from objective0 waypoint2)",
			"(at-lander general waypoint9)",
			"(visible waypoint3 waypoint9)",
			"(visible waypoint4 waypoint9)",
			"(visible waypoint5 waypoint9)",
			"(visible waypoint6 waypoint9)",
			"(channel-free general)",
		],
		runtime_objects=(
			"objective0",
			"high_res",
			"camera0",
			"camera1",
			"camera2",
			"rover0",
			"rover1",
			"rover2",
			"waypoint0",
			"waypoint1",
			"waypoint2",
			"waypoint3",
			"waypoint4",
			"waypoint5",
			"waypoint6",
			"waypoint9",
			"general",
		),
		object_types={
			"objective0": "objective",
			"high_res": "mode",
			"camera0": "camera",
			"camera1": "camera",
			"camera2": "camera",
			"rover0": "rover",
			"rover1": "rover",
			"rover2": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint2": "waypoint",
			"waypoint3": "waypoint",
			"waypoint4": "waypoint",
			"waypoint5": "waypoint",
			"waypoint6": "waypoint",
			"waypoint9": "waypoint",
			"general": "lander",
		},
		type_parent_map={
			"objective": "object",
			"mode": "object",
			"camera": "object",
			"rover": "object",
			"waypoint": "object",
			"lander": "object",
		},
		max_candidates_per_clause=4,
	)

	send_probe_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!send_probe(")
	]

	assert send_probe_chunks
	assert any(
		"on_board(camera0, rover0)" in chunk and "!helper_at(rover0, waypoint3);" in chunk
		for chunk in send_probe_chunks
	)
	assert any(
		chunk.startswith(
			"+!send_probe(OBJECTIVE, MODE) : object_type(OBJECTIVE, objective)",
		)
		and "on_board(CAMERA, ROVER)" in chunk
		for chunk in send_probe_chunks
	)


def test_compile_method_plans_preserves_navigation_support_order_before_other_compound_goals():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_prepare_sample(ROVER, SAMPLE) : prepared_sample(ROVER, SAMPLE) <-
	.print("runtime trace method flat ", "m_helper_prepare_sample_noop", "|", SAMPLE);
	true.

+!ship_sample(ROVER, SAMPLE, COMMS) : ready_sample(ROVER, SAMPLE) & comms_target(COMMS) <-
	.print("runtime trace method flat ", "m_ship_sample_constructive_1", "|", SAMPLE);
	pipeline.no_ancestor_goal(helper_at, ROVER, COMMS);
	!helper_at(ROVER, COMMS);
	pipeline.no_ancestor_goal(helper_prepare_sample, ROVER, SAMPLE);
	!helper_prepare_sample(ROVER, SAMPLE);
	!transmit_sample(ROVER, SAMPLE, COMMS).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(at rover0 waypoint0)", "(ready-sample rover0 sample0)", "(comms-target waypoint5)"],
		runtime_objects=("rover0", "waypoint0", "waypoint5", "sample0"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint5": "waypoint",
			"sample0": "sample",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
			"sample": "object",
		},
	)

	ship_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!ship_sample(")
	]

	assert any(
		"!helper_prepare_sample(ROVER, SAMPLE);" in chunk
		and "!helper_at(ROVER, COMMS);" in chunk
		and "pipeline.no_ancestor_goal(helper_prepare_sample, ROVER, SAMPLE);" in chunk
		and "pipeline.no_ancestor_goal(helper_at, ROVER, COMMS);" in chunk
		and chunk.index("!helper_at(ROVER, COMMS);") < chunk.index("!helper_prepare_sample(ROVER, SAMPLE);")
		and chunk.index(
			"pipeline.no_ancestor_goal(helper_at, ROVER, COMMS);",
		) < chunk.index("!helper_at(ROVER, COMMS);")
		and chunk.index(
			"pipeline.no_ancestor_goal(helper_prepare_sample, ROVER, SAMPLE);",
		) < chunk.index("!helper_prepare_sample(ROVER, SAMPLE);")
		for chunk in ship_chunks
	)


def test_compile_method_plans_prefers_dfa_variants_that_match_transition_literal_tokens():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!dfa_step_q1_q3_communicated_rock_data_waypoint3(WAYPOINT1) : at_rock_sample(WAYPOINT1) & object_type(WAYPOINT1, waypoint) & object_type(rover1, rover) & object_type(general, lander) & object_type(waypoint4, waypoint) & object_type(waypoint2, waypoint) & visible(waypoint4, waypoint2) & available(rover1) & channel_free(general) & pipeline.no_ancestor_goal(helper_at, rover1, waypoint4) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q3_communicated_rock_data_waypoint3_constructive_1", "|", WAYPOINT1);
	pipeline.no_ancestor_goal(helper_at, rover1, waypoint4);
	!helper_at(rover1, waypoint4);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint4, waypoint2).

+!dfa_step_q1_q3_communicated_rock_data_waypoint3(WAYPOINT1) : at_rock_sample(WAYPOINT1) & object_type(WAYPOINT1, waypoint) & object_type(rover1, rover) & object_type(general, lander) & object_type(waypoint3, waypoint) & object_type(waypoint2, waypoint) & visible(waypoint3, waypoint2) & available(rover1) & channel_free(general) & pipeline.no_ancestor_goal(helper_at, rover1, waypoint3) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q3_communicated_rock_data_waypoint3_constructive_1", "|", WAYPOINT1);
	pipeline.no_ancestor_goal(helper_at, rover1, waypoint3);
	!helper_at(rover1, waypoint3);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint3, waypoint2).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=["(at-rock-sample waypoint3)", "(visible waypoint3 waypoint2)", "(visible waypoint4 waypoint2)"],
		runtime_objects=("waypoint2", "waypoint3", "waypoint4", "rover1", "general"),
		object_types={
			"waypoint2": "waypoint",
			"waypoint3": "waypoint",
			"waypoint4": "waypoint",
			"rover1": "rover",
			"general": "lander",
		},
		type_parent_map={
			"waypoint": "object",
			"rover": "object",
			"lander": "object",
		},
	)

	dfa_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!dfa_step_q1_q3_communicated_rock_data_waypoint3(")
	]

	assert "object_type(waypoint3, waypoint)" in dfa_chunks[0]
	assert "object_type(waypoint4, waypoint)" in dfa_chunks[1]


def test_compile_method_plans_prefers_later_replay_fact_pairs_over_initial_seed_pairs():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!dfa_step_q1_q4_communicated_rock_data_waypoint6(WAYPOINT1) : at_rock_sample(WAYPOINT1) & object_type(WAYPOINT1, waypoint) & object_type(rover1, rover) & object_type(general, lander) & object_type(waypoint4, waypoint) & object_type(waypoint2, waypoint) & visible(waypoint4, waypoint2) & available(rover1) & channel_free(general) & pipeline.no_ancestor_goal(helper_at, rover1, waypoint4) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q4_communicated_rock_data_waypoint6_constructive_1", "|", WAYPOINT1);
	pipeline.no_ancestor_goal(helper_at, rover1, waypoint4);
	!helper_at(rover1, waypoint4);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint4, waypoint2).

+!dfa_step_q1_q4_communicated_rock_data_waypoint6(WAYPOINT1) : at_rock_sample(WAYPOINT1) & object_type(WAYPOINT1, waypoint) & object_type(rover1, rover) & object_type(general, lander) & object_type(waypoint3, waypoint) & object_type(waypoint2, waypoint) & visible(waypoint3, waypoint2) & available(rover1) & channel_free(general) & pipeline.no_ancestor_goal(helper_at, rover1, waypoint3) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q4_communicated_rock_data_waypoint6_constructive_1", "|", WAYPOINT1);
	pipeline.no_ancestor_goal(helper_at, rover1, waypoint3);
	!helper_at(rover1, waypoint3);
	!communicate_rock_data(rover1, general, WAYPOINT1, waypoint3, waypoint2).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(at rover1 waypoint4)",
			"(at rover1 waypoint3)",
			"(visible waypoint3 waypoint2)",
			"(visible waypoint4 waypoint2)",
		],
		runtime_objects=("waypoint2", "waypoint3", "waypoint4", "rover1", "general"),
		object_types={
			"waypoint2": "waypoint",
			"waypoint3": "waypoint",
			"waypoint4": "waypoint",
			"rover1": "rover",
			"general": "lander",
		},
		type_parent_map={
			"waypoint": "object",
			"rover": "object",
			"lander": "object",
		},
	)

	dfa_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!dfa_step_q1_q4_communicated_rock_data_waypoint6(")
	]

	assert "object_type(waypoint3, waypoint)" in dfa_chunks[0]
	assert "object_type(waypoint4, waypoint)" in dfa_chunks[1]


def test_compile_method_plans_demotes_unreachable_navigation_handoffs_after_earlier_output():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(ROVER, TARGET) : can_traverse(ROVER, PREDECESSOR, TARGET) & visible(PREDECESSOR, TARGET) & available(ROVER) & object_type(ROVER, rover) & object_type(PREDECESSOR, waypoint) & object_type(TARGET, waypoint) & PREDECESSOR \\== TARGET <-
	.print("runtime trace method flat ", "m_helper_at_constructive", "|", ROVER, "|", TARGET);
	pipeline.no_ancestor_goal(helper_at, ROVER, PREDECESSOR);
	!helper_at(ROVER, PREDECESSOR);
	!navigate(ROVER, PREDECESSOR, TARGET).

+!helper_prepare_sample(ROVER, SAMPLE_WAYPOINT) : sample_target(SAMPLE_WAYPOINT) & prepared_sample(ROVER, SAMPLE_WAYPOINT) <-
	.print("runtime trace method flat ", "m_helper_prepare_sample_constructive", "|", ROVER, "|", SAMPLE_WAYPOINT);
	pipeline.no_ancestor_goal(helper_at, ROVER, SAMPLE_WAYPOINT);
	!helper_at(ROVER, SAMPLE_WAYPOINT);
	!collect_sample(ROVER, SAMPLE_WAYPOINT).

+!ship_sample(SAMPLE_WAYPOINT) : sample_target(SAMPLE_WAYPOINT) & object_type(rover0, rover) & comms_target(waypoint1) <-
	.print("runtime trace method flat ", "m_ship_sample_constructive_1", "|", SAMPLE_WAYPOINT);
	pipeline.no_ancestor_goal(helper_prepare_sample, rover0, SAMPLE_WAYPOINT);
	!helper_prepare_sample(rover0, SAMPLE_WAYPOINT);
	pipeline.no_ancestor_goal(helper_at, rover0, waypoint1);
	!helper_at(rover0, waypoint1);
	!transmit_sample(rover0, SAMPLE_WAYPOINT, waypoint1).

+!ship_sample(SAMPLE_WAYPOINT) : sample_target(SAMPLE_WAYPOINT) & object_type(rover0, rover) & comms_target(waypoint6) <-
	.print("runtime trace method flat ", "m_ship_sample_constructive_1", "|", SAMPLE_WAYPOINT);
	pipeline.no_ancestor_goal(helper_prepare_sample, rover0, SAMPLE_WAYPOINT);
	!helper_prepare_sample(rover0, SAMPLE_WAYPOINT);
	pipeline.no_ancestor_goal(helper_at, rover0, waypoint6);
	!helper_at(rover0, waypoint6);
	!transmit_sample(rover0, SAMPLE_WAYPOINT, waypoint6).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint0)",
			"(sample-target waypoint4)",
			"(prepared-sample rover0 waypoint4)",
			"(comms-target waypoint1)",
			"(comms-target waypoint6)",
			"(can-traverse rover0 waypoint0 waypoint4)",
			"(can-traverse rover0 waypoint4 waypoint6)",
			"(visible waypoint0 waypoint4)",
			"(visible waypoint4 waypoint6)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1", "waypoint4", "waypoint6"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint4": "waypoint",
			"waypoint6": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	ship_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!ship_sample(")
	]

	assert "!helper_at(rover0, waypoint6);" in ship_chunks[0]
	assert "!helper_at(rover0, waypoint1);" in ship_chunks[1]


def test_compile_method_plans_demotes_residual_generic_witness_chunks_below_grounded_variants():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!dfa_step_q1_q6_communicated_image_data_objective1_high_res(objective1, high_res) : object_type(objective1, objective) & object_type(high_res, mode) & object_type(witness_rover_1, rover) & object_type(general, lander) & object_type(waypoint0, waypoint) & object_type(waypoint2, waypoint) & have_image(witness_rover_1, objective1, high_res) & available(witness_rover_1) & at_lander(general, waypoint2) & visible(waypoint0, waypoint2) & pipeline.no_ancestor_goal(helper_have_image, witness_rover_1, objective1, high_res) & pipeline.no_ancestor_goal(helper_at, witness_rover_1, waypoint0) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q6_communicated_image_data_objective1_high_res_constructive_1", "|", objective1, "|", high_res);
	pipeline.no_ancestor_goal(helper_have_image, witness_rover_1, objective1, high_res);
	!helper_have_image(witness_rover_1, objective1, high_res);
	pipeline.no_ancestor_goal(helper_at, witness_rover_1, waypoint0);
	!helper_at(witness_rover_1, waypoint0);
	!communicate_image_data(witness_rover_1, general, objective1, high_res, waypoint0, waypoint2).

+!dfa_step_q1_q6_communicated_image_data_objective1_high_res(objective1, high_res) : object_type(objective1, objective) & object_type(high_res, mode) & object_type(witness_rover_1, rover) & object_type(witness_lander_1, lander) & object_type(witness_waypoint_2, waypoint) & object_type(witness_waypoint_1, waypoint) & have_image(witness_rover_1, objective1, high_res) & available(witness_rover_1) & at_lander(witness_lander_1, witness_waypoint_1) & visible(witness_waypoint_2, witness_waypoint_1) & pipeline.no_ancestor_goal(helper_have_image, witness_rover_1, objective1, high_res) & pipeline.no_ancestor_goal(helper_at, witness_rover_1, witness_waypoint_2) <-
	.print("runtime trace method flat ", "m_dfa_step_q1_q6_communicated_image_data_objective1_high_res_constructive_1", "|", objective1, "|", high_res);
	pipeline.no_ancestor_goal(helper_have_image, witness_rover_1, objective1, high_res);
	!helper_have_image(witness_rover_1, objective1, high_res);
	pipeline.no_ancestor_goal(helper_at, witness_rover_1, witness_waypoint_2);
	!helper_at(witness_rover_1, witness_waypoint_2);
	!communicate_image_data(witness_rover_1, witness_lander_1, objective1, high_res, witness_waypoint_2, witness_waypoint_1).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(have-image rover2 objective1 high_res)",
			"(available rover2)",
			"(at-lander general waypoint2)",
			"(visible waypoint0 waypoint2)",
			"(visible waypoint4 waypoint2)",
		],
		runtime_objects=("objective1", "high_res", "rover2", "general", "waypoint0", "waypoint2", "waypoint4"),
		object_types={
			"objective1": "objective",
			"high_res": "mode",
			"rover2": "rover",
			"general": "lander",
			"waypoint0": "waypoint",
			"waypoint2": "waypoint",
			"waypoint4": "waypoint",
		},
		type_parent_map={
			"objective": "object",
			"mode": "object",
			"rover": "object",
			"lander": "object",
			"waypoint": "object",
		},
	)

	dfa_chunks = [
		chunk
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!dfa_step_q1_q6_communicated_image_data_objective1_high_res(")
	]

	assert "object_type(waypoint0, waypoint)" in dfa_chunks[0]
	assert "witness_waypoint_2" in dfa_chunks[1]


def test_compile_method_plans_adds_noop_prefix_context_to_grounded_recursive_chunk():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(rover0, waypoint3) : can_traverse(rover0, waypoint0, waypoint3) & visible(waypoint0, waypoint3) & available(rover0) & object_type(rover0, rover) & object_type(waypoint3, waypoint) & object_type(waypoint0, waypoint) & waypoint0 \\== waypoint3 <-
	.print("runtime trace method flat ", "m_helper_at_constructive_1", "|", rover0, "|", waypoint3);
	pipeline.no_ancestor_goal(helper_at, rover0, waypoint0);
	!helper_at(rover0, waypoint0);
	!navigate(rover0, waypoint0, waypoint3).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint1)",
			"(can-traverse rover0 waypoint0 waypoint3)",
			"(visible waypoint0 waypoint3)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1", "waypoint3"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint3": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	guarded_head = (
		'+!helper_at(rover0, waypoint3) : can_traverse(rover0, waypoint0, waypoint3)'
		' & visible(waypoint0, waypoint3) & available(rover0) & object_type(rover0, rover)'
		' & object_type(waypoint3, waypoint) & object_type(waypoint0, waypoint)'
		' & waypoint0 \\== waypoint3 & at(rover0, waypoint0)'
		' & pipeline.no_ancestor_goal(helper_at, rover0, waypoint0) <-'
	)
	unguarded_head = (
		'+!helper_at(rover0, waypoint3) : can_traverse(rover0, waypoint0, waypoint3)'
		' & visible(waypoint0, waypoint3) & available(rover0) & object_type(rover0, rover)'
		' & object_type(waypoint3, waypoint) & object_type(waypoint0, waypoint)'
		' & waypoint0 \\== waypoint3 & pipeline.no_ancestor_goal(helper_at, rover0, waypoint0) <-'
	)

	assert guarded_head.removesuffix(" <-") in rewritten
	assert unguarded_head not in rewritten


def test_compile_method_plans_avoids_conflicting_self_recursive_noop_state_duplication():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!helper_at(ROVER, LOCATION) : at(ROVER, LOCATION) & object_type(ROVER, rover) & object_type(LOCATION, waypoint) <-
	.print("runtime trace method flat ", "m_helper_at_noop", "|", ROVER, "|", LOCATION);
	true.

+!helper_at(rover0, waypoint3) : can_traverse(rover0, waypoint0, waypoint3) & visible(waypoint0, waypoint3) & available(rover0) & object_type(rover0, rover) & object_type(waypoint3, waypoint) & object_type(waypoint0, waypoint) & waypoint0 \\== waypoint3 & at(rover0, waypoint1) <-
	.print("runtime trace method flat ", "m_helper_at_constructive_conflict", "|", rover0, "|", waypoint3);
	pipeline.no_ancestor_goal(helper_at, rover0, waypoint0);
	!helper_at(rover0, waypoint0);
	!navigate(rover0, waypoint0, waypoint3).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(available rover0)",
			"(at rover0 waypoint1)",
			"(can-traverse rover0 waypoint0 waypoint3)",
			"(visible waypoint0 waypoint3)",
		],
		runtime_objects=("rover0", "waypoint0", "waypoint1", "waypoint3"),
		object_types={
			"rover0": "rover",
			"waypoint0": "waypoint",
			"waypoint1": "waypoint",
			"waypoint3": "waypoint",
		},
		type_parent_map={
			"rover": "object",
			"waypoint": "object",
		},
	)

	assert not any(
		lines
		and lines[0].startswith("+!helper_at(rover0, waypoint3)")
		and "at(rover0, waypoint1)" in lines[0]
		and "at(rover0, waypoint0)" in lines[0]
		for lines in (
			chunk.splitlines()
			for chunk in rewritten.split("\n\n")
		)
	)


def test_compile_method_plans_preserves_ordered_body_sequence_for_transport_like_delivery():
	lowering = ASLMethodLowering()
	agentspeak_code = """
/* HTN Method Plans */
+!get_to(VEHICLE, LOCATION) : at(VEHICLE, LOCATION) & object_type(VEHICLE, vehicle) & object_type(LOCATION, location) <-
	.print("runtime trace method flat ", "m-i-am-there", "|", VEHICLE, "|", LOCATION);
	true.

+!deliver(PACKAGE, LOCATION1) : at(PACKAGE, LOCATION2) & object_type(PACKAGE, package) & object_type(LOCATION1, location) & object_type(LOCATION2, location) & object_type(VEHICLE, vehicle) & capacity(VEHICLE, CAPACITY_NUMBER2) <-
	.print("runtime trace method flat ", "m-deliver", "|", PACKAGE, "|", LOCATION1);
	!get_to(VEHICLE, LOCATION2);
	!load(VEHICLE, LOCATION2, PACKAGE);
	!get_to(VEHICLE, LOCATION1);
	!unload(VEHICLE, LOCATION1, PACKAGE).

/* DFA Transition Wrappers */
""".strip() + "\n"

	rewritten = lowering.compile_method_plans(
		agentspeak_code,
		seed_facts=[
			"(at package-4 city-loc-1)",
			"(at truck-0 city-loc-1)",
			"(capacity truck-0 capacity-2)",
		],
		runtime_objects=("package-4", "truck-0", "city-loc-1", "city-loc-4", "capacity-2"),
		object_types={
			"package-4": "package",
			"truck-0": "vehicle",
			"city-loc-1": "location",
			"city-loc-4": "location",
			"capacity-2": "capacity_number",
		},
		type_parent_map={
			"package": "object",
			"vehicle": "object",
			"location": "object",
			"capacity_number": "object",
		},
	)

	deliver_chunks = [
		chunk.splitlines()
		for chunk in rewritten.split("\n\n")
		if chunk.startswith("+!deliver(")
	]

	assert deliver_chunks
	for lines in deliver_chunks:
		assert lines[2].startswith("\t!get_to(")
		assert lines[3].startswith("\t!load(")
		assert lines[4].startswith("\t!get_to(")
		assert lines[5].startswith("\t!unload(")
		assert lines[2].endswith(";")
		assert lines[3].endswith(";")
		assert lines[4].endswith(";")
		assert lines[5].endswith(".")


def test_preserve_chunk_growth_budget_keeps_all_base_chunks_and_caps_extras():
	base_chunks = ["A", "B", "C"]
	expanded_chunks = ["X", "A", "Y", "B", "Z", "C", "W"]

	assert ASLMethodLowering._preserve_chunk_growth_budget(
		base_chunks,
		expanded_chunks,
		max_total_specialised_chunks=2,
	) == ["X", "A", "Y", "B", "C"]


def test_preserve_chunk_growth_budget_distributes_extra_chunks_across_base_chunks():
	base_chunks = ["A", "B", "C"]
	expanded_chunks = ["X1", "X2", "A", "Y", "B", "Z", "C"]

	assert ASLMethodLowering._preserve_chunk_growth_budget(
		base_chunks,
		expanded_chunks,
		max_total_specialised_chunks=2,
	) == ["X1", "A", "Y", "B", "C"]


def test_chunk_is_noop_method_plan_accepts_parameterised_noop_calls():
	assert ASLMethodLowering._chunk_is_noop_method_plan(
		(
			'\t.print("runtime trace method flat ", "m-noop", "|", VEHICLE, "|", LOCATION);',
			"\t!noop(VEHICLE, LOCATION).",
		),
	)
