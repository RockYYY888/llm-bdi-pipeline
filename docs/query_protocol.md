# Query Protocol

Write one single sentence in this form:

`Using <typed objects actually referenced by the requested tasks>, complete the tasks task_1(arg1, ...), task_2(arg1, ...), ... .`

Rules:
- Use the exact object identifiers from the benchmark vocabulary.
- Write every requested declared task explicitly as `task(arg, ...)`.
- Keep the task order when the intended task network is ordered.
- Do not mention `problem.hddl`, `:init`, `:goal`, hidden state facts, methods, repairs, or decomposition advice.

Example:

`Using blocks b4, b2, b1, and b3, complete the tasks do_put_on(b4, b2), do_put_on(b1, b4), and do_put_on(b3, b1).`
