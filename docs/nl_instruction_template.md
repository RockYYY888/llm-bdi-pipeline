# Natural-Language Query Protocol

This document defines the current natural-language query style for the pipeline.
It is a protocol document, not a generic LTL tutorial.

## 1. Purpose

The query text has two jobs:

1. Stage 1 uses it to derive predicate-grounded LTLf.
2. Stage 3 uses any explicitly named declared tasks as the preferred HTN skeleton.

The query must therefore be compact, explicit, and free of hidden problem-instance semantics.

## 2. Recommended Query Form

For reliable Stage 1 interpretation and Stage 3 synthesis, use one single sentence in this form:

```text
Using <typed query-referenced objects>, complete the tasks task_1(arg1, ...), task_2(arg1, ...), and task_3(arg1, ...).
```

## 3. Required Properties

Every query should satisfy all of the following.

1. Single sentence.
2. Uses the exact object identifiers that appear in the domain/problem vocabulary.
3. Uses explicit task-invocation syntax for declared tasks.
4. Includes the typed objects referenced by the root task network. If the root task network
   still contains variables, include the typed candidate objects needed to ground them.
5. Does not mention `problem.hddl`, `:init`, `:goal`, benchmark ids, or hidden initial-state facts.
6. Does not describe desired methods, repairs, or decomposition strategies directly.

## 4. Why This Form Is the Current Default

This is not cosmetic. It matches the current pipeline boundary.

1. Stage 1 still produces predicate-grounded LTLf, not task atoms.
2. Stage 3 treats query-mentioned declared tasks as the preferred high-level method skeleton.
3. Static facts such as capabilities, topology, visibility, and equipment should stay in method
   context, not be described as extra user goals unless they are genuine targets.
4. The format prevents `problem.hddl` semantics from leaking into Stage 1 and Stage 3 while still
   exposing the task structure needed for benchmark-backed evaluation.

## 5. Benchmark Query Rule

For official benchmark-backed acceptance:

1. Each `problem.hddl` instance maps to exactly one query.
2. The query is reverse-generated from:
   - the problem's root HTN tasks
   - the minimal typed object inventory justified by those task invocations
3. The query is not manually authored per problem instance.
4. `problem.hddl` remains reserved for:
   - Stage 6 runtime initialisation
   - Stage 7 official verification

## 5.1 Materialised Benchmark Query Manifest

For reproducibility, the benchmark queries are not left as implicit runtime derivations.

1. The checked-in materialisation lives at
   `src/benchmark_data/official_problem_queries.json`.
2. The canonical generator and loader live at
   `src/utils/benchmark_query_manifest.py`.
3. The manifest is produced deterministically from:
   - the problem root HTN task network
   - the query-referenced typed object inventory, with full problem inventory only when root
     task variables require grounding candidates
   - the query protocol in this document
4. The acceptance harness reads the manifest rather than regenerating query text inline.
5. Unit tests verify that every manifest entry still matches the canonical reverse-generation
   rule, so the stored data remains auditable rather than hand-edited folklore.

This separation is deliberate: the document defines the protocol, the manifest stores the
versioned benchmark instances, and the tests verify that the stored instances are justified by
the canonical rule.

## 6. Current Examples

Blocksworld:

```text
Using blocks b4, b2, b1, and b3, complete the tasks do_put_on(b4, b2), do_put_on(b1, b4), and do_put_on(b3, b1).
```

Marsrover:

```text
Using waypoints waypoint2 and waypoint3, objective objective1, and mode high_res, complete the tasks get_soil_data(waypoint2), get_rock_data(waypoint3), and get_image_data(objective1, high_res).
```

## 7. Anti-Patterns

Avoid these query styles.

1. Benchmark leakage:

```text
Solve pfile01 using the official initial state.
```

2. Hidden initial-state assumptions:

```text
Since rover0 is already calibrated and visible to the lander, just transmit the image.
```

3. Multi-sentence or narrative descriptions:

```text
First get the soil sample. Then maybe send it. Also keep the rover safe.
```

4. Method-authoring instructions instead of task requests:

```text
Create a helper task for emptying the store before sampling.
```

## 8. Scope Notes

This document governs query writing only. It does not replace:

- `README.md` for the pipeline overview
- `PIPELINE_ASSUMPTIONS.md` for formal scope and validation boundaries
- `src/stage1_interpretation/prompts.py` for the exact Stage 1 output contract
- `src/stage3_method_synthesis/htn_prompts.py` for the exact Stage 3 synthesis contract
- `src/benchmark_data/official_problem_queries.json` for the benchmark query instance inventory
