from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

import utils.benchmark_query_dataset as benchmark_query_dataset


def test_load_problem_query_cases_reads_explicit_json_binding_only(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	problem_dir = tmp_path / "blocksworld" / "problems"
	problem_dir.mkdir(parents=True)
	(problem_dir / "p01.hddl").write_text("(define (problem p01))")
	dataset_path = tmp_path / "benchmark_queries.json"
	dataset_path.write_text(
		json.dumps(
			{
				"version": 5,
				"dataset_kind": "stored_benchmark_queries",
				"query_protocol_document": "docs/query_protocol.md",
				"generator": "test",
				"domains": {
					"blocksworld": {
						"cases": {
							"query_1": {
								"instruction": "complete the tasks do_put_on(b4, b2).",
								"problem_file": "p01.hddl",
							},
						},
					},
				},
			},
			indent=2,
		),
	)
	monkeypatch.setattr(
		benchmark_query_dataset,
		"_infer_domain_key_from_problem_dir",
		lambda _: "blocksworld",
	)
	monkeypatch.setattr(
		benchmark_query_dataset,
		"build_stored_case_from_problem",
		lambda _path: (_ for _ in ()).throw(AssertionError("runtime must not rebuild cases")),
	)

	cases = benchmark_query_dataset.load_problem_query_cases(
		problem_dir,
		dataset_path=dataset_path,
	)

	assert cases == {
		"query_1": {
			"instruction": "complete the tasks do_put_on(b4, b2).",
			"problem_file": str((problem_dir / "p01.hddl").resolve()),
		},
	}


def test_load_problem_query_cases_rejects_legacy_string_cases(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	problem_dir = tmp_path / "transport" / "problems"
	problem_dir.mkdir(parents=True)
	(problem_dir / "pfile01.hddl").write_text("(define (problem pfile01))")
	dataset_path = tmp_path / "benchmark_queries.json"
	dataset_path.write_text(
		json.dumps(
			{
				"version": 5,
				"dataset_kind": "stored_benchmark_queries",
				"query_protocol_document": "docs/query_protocol.md",
				"generator": "test",
				"domains": {
					"transport": {
						"cases": {
							"query_1": "deliver(package-0, city-loc-0)",
						},
					},
				},
			},
			indent=2,
		),
	)
	monkeypatch.setattr(
		benchmark_query_dataset,
		"_infer_domain_key_from_problem_dir",
		lambda _: "transport",
	)

	with pytest.raises(ValueError, match="must be an object record"):
		benchmark_query_dataset.load_problem_query_cases(
			problem_dir,
			dataset_path=dataset_path,
		)


def test_load_problem_query_cases_rejects_missing_problem_file(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	problem_dir = tmp_path / "satellite" / "problems"
	problem_dir.mkdir(parents=True)
	dataset_path = tmp_path / "benchmark_queries.json"
	dataset_path.write_text(
		json.dumps(
			{
				"version": 5,
				"dataset_kind": "stored_benchmark_queries",
				"query_protocol_document": "docs/query_protocol.md",
				"generator": "test",
				"domains": {
					"satellite": {
						"cases": {
							"query_1": {
								"instruction": "do_observation(Phenomenon4, thermograph0)",
							},
						},
					},
				},
			},
			indent=2,
		),
	)
	monkeypatch.setattr(
		benchmark_query_dataset,
		"_infer_domain_key_from_problem_dir",
		lambda _: "satellite",
	)

	with pytest.raises(ValueError, match='missing a non-empty "problem_file"'):
		benchmark_query_dataset.load_problem_query_cases(
			problem_dir,
			dataset_path=dataset_path,
		)


def test_load_problem_query_cases_rejects_problem_path_escape(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	problem_dir = tmp_path / "marsrover" / "problems"
	problem_dir.mkdir(parents=True)
	(tmp_path / "pfile01.hddl").write_text("(define (problem pfile01))")
	dataset_path = tmp_path / "benchmark_queries.json"
	dataset_path.write_text(
		json.dumps(
			{
				"version": 5,
				"dataset_kind": "stored_benchmark_queries",
				"query_protocol_document": "docs/query_protocol.md",
				"generator": "test",
				"domains": {
					"marsrover": {
						"cases": {
							"query_1": {
								"instruction": "get_soil_data(waypoint2)",
								"problem_file": "../pfile01.hddl",
							},
						},
					},
				},
			},
			indent=2,
		),
	)
	monkeypatch.setattr(
		benchmark_query_dataset,
		"_infer_domain_key_from_problem_dir",
		lambda _: "marsrover",
	)

	with pytest.raises(ValueError, match='uses illegal "problem_file" path'):
		benchmark_query_dataset.load_problem_query_cases(
			problem_dir,
			dataset_path=dataset_path,
		)
