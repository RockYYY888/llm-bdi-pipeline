"""
Compatibility wrapper for the old benchmark query manifest naming.

The canonical interface is now `utils.benchmark_query_dataset`, which exposes
the stored benchmark queries as a dataset rather than as an implicitly magical
"manifest". Keep these aliases so older scripts do not break.
"""

from __future__ import annotations

from utils.benchmark_query_dataset import (
	DEFAULT_BENCHMARK_QUERY_DATASET_PATH,
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PATTERNS,
	DEFAULT_BENCHMARK_QUERY_DOMAIN_PROBLEM_DIRS,
	build_benchmark_query_dataset,
	build_case_from_problem,
	load_benchmark_query_dataset,
	load_problem_query_cases,
	query_referenced_problem_objects,
	serialise_nl_list,
	serialise_task_clause_sequence,
	task_invocation_to_query_clause,
	typed_object_phrase,
	typed_object_phrase_for_objects,
	write_benchmark_query_dataset,
)

DEFAULT_BENCHMARK_QUERY_MANIFEST_PATH = DEFAULT_BENCHMARK_QUERY_DATASET_PATH


def build_benchmark_query_manifest():
	return build_benchmark_query_dataset()


def write_benchmark_query_manifest(manifest_path=None):
	return write_benchmark_query_dataset(manifest_path)


def load_benchmark_query_manifest(manifest_path=None):
	return load_benchmark_query_dataset(manifest_path)
