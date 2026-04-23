# Query Protocol

The query protocol describes how a human user should write one natural-language
task query.

User-facing query rules:

- Natural language is allowed.
- The query must be one sentence, even when it contains several task references.
- The query should state the intended task-level behaviour.
- The query may use ordinary language or declared task names from the domain.
- The query may mention task-relevant objects when they are known.
- The query does not need to enumerate the full object inventory of the current instance.
- The query does not need to define the initial state or the complete environment.
- The query does not need to restate the instance goal condition.
- Temporal wording such as `first`, `then`, `before`, `after`, `until`, `always`, and `eventually` may be used when it is part of the intended task request.
- Temporal wording expresses desired relations among tasks; it should not be written as a procedural decomposition.
- The query text must not mention decomposition advice, methods, planner configuration, solver choices, or execution-level implementation guidance.

Assumed context:

- The query is interpreted relative to an existing planning domain.
- When instance-level grounding is needed, the relevant instance supplies the available objects, initial state, and goal condition.
- These contextual elements do not need to be repeated in the query.

Recommended writing rules:

- Use exact object identifiers when they are known.
- Naming declared task calls explicitly still improves grounding stability.
- Keep temporal wording explicit when order matters.

Canonical example:

`Using blocks b4, b2, b1, and b3, complete the tasks do_put_on(b4, b2), then do_put_on(b1, b4), then do_put_on(b3, b1).`

Free-form example:

`First put block b4 on block b2, then stack b1 on top of b4, and finally place b3 on b1.`
