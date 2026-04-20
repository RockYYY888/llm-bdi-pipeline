# Query Protocol

The online Jason path consumes exactly one natural-language query string.

User-facing query rules:

- Natural language is allowed.
- Task invocations can be explicit or implicit.
- Temporal wording such as `first`, `then`, `before`, `after`, `until`, `always`, and `eventually` is interpreted by the large-language-model grounding step.
- The query text does not need to enumerate the full problem object inventory.
- The query text does not need to verbalize the benchmark `:goal`.
- The query text must not mention decomposition advice, methods, or planner internals.

System-provided grounding context:

- The online runtime may provide callable grounded task signatures from the current domain library.
- The online runtime may provide grounded problem objects from the current `problem.hddl`.
- These inventories are runtime grounding context. They are not required text in the user query protocol.

Benchmark-canonical query records:

- `src/benchmark_data/benchmark_queries.json` is the single source of truth.
- Each stored `query_id` remains a stable benchmark handle for reruns such as `query_7`.
- Each stored record must bind:
  - `instruction`: the canonical natural-language query
  - `problem_file`: the benchmark problem file used for execution and official verification
- The canonical benchmark query semantics are task-centric:
  - they express the root task request and its temporal order
  - they do not explicitly verbalize benchmark `:goal`
  - official verification still runs against the bound original `problem_file`

Recommended writing rules:

- Use the exact benchmark object identifiers when you know them.
- Naming declared task calls explicitly still improves grounding stability.
- Keep temporal wording explicit when order matters.

Canonical example:

`Using blocks b4, b2, b1, and b3, complete the tasks do_put_on(b4, b2), then do_put_on(b1, b4), then do_put_on(b3, b1).`

Free-form example:

`First put block b4 on block b2, then stack b1 on top of b4, and finally place b3 on b1.`
