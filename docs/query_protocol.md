# Query Protocol

The online Jason path no longer requires the benchmark-canonical single-sentence template.

Accepted query style:

- natural language is allowed
- task invocations can be explicit or implicit
- temporal wording such as `first`, `then`, `before`, `after`, `until`, `always`, and `eventually` is interpreted by the large-language-model grounding step
- all grounded objects must still come from the current `problem.hddl` object inventory

Recommended rules:

- Use the exact object identifiers from the benchmark vocabulary.
- If you already know the intended declared task calls, naming them explicitly still improves grounding stability.
- Keep temporal wording explicit when order matters.
- Do not mention decomposition advice, methods, or planner internals.

Still-valid canonical example:

`Using blocks b4, b2, b1, and b3, first do_put_on(b4, b2), then do_put_on(b1, b4), then do_put_on(b3, b1).`

Free-form example:

`First put block b4 on block b2, then stack b1 on top of b4, and finally place b3 on b1.`
