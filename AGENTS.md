# Project Instructions

This repository is a PDDL-only, domain-level AgentSpeak(L) plan-library
pipeline. The current research target changed on 2026-07-03.

## Current Architecture

Do not build a universal generalized planner inside this repository. Do not
route a domain to a backend by assigning it to a prior paper taxonomy class.

## Maintained Research Design Documents

Three pre-paper documents are normative and must remain synchronized with the
implementation and manuscript:

- `docs/research_pipeline_decisions.md` records architecture, evidence/compiler
  boundaries, benchmark scope, certificate requirements, supported fragments,
  and explicit limitations. Update it in the same commit whenever a change
  alters a research-facing claim.
- `docs/input_design.md` records the complete natural-language-to-parametric-
  LTLf contract, JSON artifacts, propositionization, real `ltlf2dfa`/MONA DFA
  construction, benchmark generation, and independent DFA trace-validation
  semantics. Update it in the same commit whenever an Input schema, parameter
  semantic, DFA capability, query-wrapper contract, or temporal evaluation
  oracle changes.
- `docs/aamas_paper_narrative_outline.md` is the canonical AAMAS 2027 manuscript
  narrative, section contract, claim boundary, result-insertion contract, and
  page-budget plan. Every paper update must follow this outline. When a better
  writing strategy is adopted, update the outline in the same commit as the
  manuscript rather than allowing the paper and plan to diverge.

The canonical query semantics are typed, externally bound, parametric LTLf.
For example, `phi(X,Y)` is compiled once and invoked under an assignment such
as `X=b1, Y=b2`; it is neither existential nor universal quantification. PDDL
objects must not be exhaustively grounded before DFA construction. A grounded
formula is permitted only as a per-invocation validation projection. Do not
claim first-order quantified LTLf unless a separate formal implementation and
evaluation are added.

The current architecture has two connected flows across five named modules:
the Typed Temporal Input Module, Evidence Module, Validated Policy-Lifting
Compiler, Temporal Query Compiler, and Execution Validation Module. Do not call
these Layer A/B/C. The complete framework is named **GP2PL** (Generalized
Planning to Plan Libraries); GP2PL is not an alias for any one module.

The manuscript must present GP2PL as a theoretical representation-compilation
framework rather than as a software architecture. Lead with the typed temporal
input contract, normalized evidence relation, certified candidate language,
feasible-library optimization, conditional completion summaries, and
preservation-safe DFA guard composition. The frozen translation model, MOOSE,
Clingo, AgentSpeak(L), and Jason are concrete instantiations of those interfaces.
Keep internal paths, command-line options, hashes, worker scheduling, and class
names in the technical or code-and-data appendix.

The canonical technical appendix is
`latex_code/aamas_method_paper/appendix/technical_appendix.tex`. Public reproducibility
evidence lives under `paper_artifacts/temporal_goal_benchmark/v1`,
`paper_artifacts/temporal_semantic_conformance/v1`, and
`paper_artifacts/gp2pl_evaluation/v1`. GP2PL code uses Apache-2.0 and original
GP2PL data uses CC BY 4.0; third-party PDDL and MOOSE files retain upstream
terms and must be fetched at pinned revisions. The camera-ready public repository
is `https://github.com/daidaibunny/gp2pl`, but anonymous AAMAS paper and appendix
PDFs must not expose that identifying URL. Keep the main-paper artifact statement
behind the camera-ready-only conditional.

```text
PDDL domain + train split
-> external generalized-planning evidence backend
-> atomic minimal literal module synthesis
-> compact lifted atomic AgentSpeak(L) library

controlled natural-language request + typed parameters
-> frozen prompt + deterministic input validation
-> validated lifted LTLf JSON
-> real ltlf2dfa/MONA DFA construction
-> conjunctive guard-transition DFA validation
-> append +!g_query wrapper plans to the same domain library
```

There must be exactly one maintained AgentSpeak(L) library per domain. New user
queries append new top-level goals such as `g_query_17` to that library.
The canonical maintained location is:

```text
artifacts/domain_libraries/<domain>/plan_library.json
artifacts/domain_libraries/<domain>/plan_library.asl
artifacts/domain_libraries/<domain>/artifact_metadata.json
```

The main CLI must reject non-canonical output roots and non-canonical append
input files. Audit scripts may write snapshot artifacts, but those snapshots are
not the maintained domain library.

## Atomic Minimal Literal Module Library

- Learn or import evidence for lifted atomic predicate/literal targets, for example
  `+!on(X,Y)`, `+!clear(X)`, `+!at(P,L)`, or `+!served(P)`.
- MOOSE is the first backend candidate for positive singleton predicate goals
  because its paper method explicitly decomposes training problems into
  singleton goal conditions and learns lifted rules by goal regression.
- MOOSE readable artifacts produced by `policy <model> --dump-policy` are
  consumed through `src/domain_level_planning/moose_policy_adapter.py`.
- Raw MOOSE singleton macro policies are evidence, not automatically the final
  domain-level library. When `--validated-policy-lifting` is used, the compiler
  validates MOOSE policy-rule macros against PDDL schemas, lifts their variables,
  preserves complete validated policy-rule branches, and uses MOOSE seed
  predicates plus PDDL action schemas to synthesize schema-augmented recursive
  modules when closure requires internal producible fluents.
  `--minimal-modules` and `--post-moose-recursive` remain only as deprecated
  compatibility aliases.
  MOOSE seed predicates are not assumed to include all required atomic modules.
  Every declared predicate that appears in a positive PDDL add effect enters
  the same schema closure, including when the evidence contains only numeric
  goals. For example, if MOOSE only saw singleton `on(X,Y)` goals in Blocks,
  `clear(X)`, `holding(X)`, `handempty`, and `ontable(X)` still enter the
  library; if numeric evidence only targeted a resource function, producible
  predicates such as `position(X)` still enter through their PDDL producers.
  Static predicates remain context only. PDDL typing is compiled into the
  reserved static sort metadata predicate
  `obj_tp(Object, Type)` when action schemas require subtype-safe binding. The
  final ASL must not emit domain-specific `type_*` guards, and `obj_tp/2` must
  remain context-only metadata that is never used as a subgoal or primitive
  action.
- Validated policy lifting with schema-augmented recursive module synthesis must
  use a solver-backed branch selector rather than local hand-written pruning
  when claiming paper-quality compactness. Prefer Clingo/ASP for the first
  implementation because Learning Sketches uses ASP-style synthesis and because
  our constraints are relational:
  selected branches must cover candidate evidence, preserve required fallback
  branch kinds, satisfy schema executability, keep producer modules emitted for
  internal closure predicates, avoid synthetic goals, and minimize branch count,
  context count, and body cost within the generated candidate space.
- The Clingo/ASP selector is mandatory for `--validated-policy-lifting`; do not
  add a silent local-pruning fallback. The selector treats each generated branch
  as an evidence obligation. A selected branch may cover another branch only
  when it has the same lifted trigger, no stronger context, and an equivalent or
  explicitly recursive body coverage relation. This is meant to remove
  redundant sibling branches while preserving necessary cases such as
  already-true, direct producer, and recursive preparation branches.
- Validated provider macros and schema-derived candidates must enter the same
  Clingo solve. Do not select schema branches first and append MOOSE macros
  afterward. Recursive branches require a non-negative relational ranking
  certificate, and resource cleanup requires a causal keyed-capacity invariant.
- MOOSE is not claimed to solve interacting conjunctive goals directly.
  Blocks-style goals such as `on(X,Y) & on(Y,Z)` require the temporal wrapper or
  another structural controller.
- KR 2025 learner-policies-from-examples, D2L, learner-sketches, and
  h-policy-learner remain audited fallback or comparison backends. A backend
  artifact counts as a current atomic-library backend only after:

```text
parse -> LiftedPolicyProgram -> verified atomic action/subgoal binding -> ASL compilation -> held-out validation
```

At present, MOOSE readable-policy artifacts are the only implemented
paper-backend-to-atomic-ASL compiler path. Do not classify whole domains by
compiler outcome. The implemented structural labels are plan-template-level:
`already_true_plan_template` for empty bodies, `action_only_plan_template` for
bodies containing only primitive PDDL actions, and
`subgoal_decomposed_plan_template` for bodies containing internal achievement
subgoals such as `!clear(Y)`. A domain library may contain several template
kinds at once, so metadata reports `library_profile` and
`plan_template_kind_counts` only as diagnostics, not as a domain taxonomy or
routing rule.

Do not refer to the current method as Layer A, Layer B, or Layer C. Use
`atomic minimal literal module library` for the compact low-level artifact and
`temporal goal append` for query-specific DFA wrappers.

- Negative literal templates are not supported unless a backend artifact gives a
  validated implementation. Reject them with a structured diagnostic instead of
  inventing synthetic subgoals.

## Temporal Input and Goal Append

- The Typed Temporal Input Module translates controlled natural-language
  requests into typed, externally bound parametric LTLf JSON under a fixed
  eight-key payload contract. Its frozen prompt, PDDL vocabulary/arity/type
  checks, and benchmark protocol are GP2PL contributions. Exact gold/predicted
  DFA-language equivalence is an evaluation oracle, not an online requirement.
- Temporal append consumes the persisted validated LTLf artifact and must not
  rerun the language model during query compilation.
- A query with no temporal operator is an ordinary achievement request and
  remains atemporal in the persisted artifact. Before the common DFA path,
  embed its completion semantics as `F(formula)`; preserve a TEG unchanged.
  Do not add a separate linear or non-DFA achievement path.
- The historical LTLf-to-DFA and logger code may be restored and refactored, but
  it must be PDDL-only and must not reintroduce HDDL, HTN, or legacy event-to-
  fluent mappings.
- LTLf formulas must be converted through the real `ltlf2dfa` package and a
  real MONA binary. Do not restore an ordered-sequence or linear-body fast path.
- Benchmark version 1 supports the explicit `F`, `X`, `U`, conjunction, and
  literal-negation fragment. Do not describe this as unrestricted LTLf.
- Every relevant DFA transition guard is interpreted as one query-local guard
  block. A guard transition is the set of literals on one MONA/ltlf2dfa transition
  guard; for example `on(X,Y) & not clear(Z)` is one block with positive
  achievement `on(X,Y)` and negative context guard `not clear(Z)`.
- Guard blocks may contain conjunction and negation only. Reject disjunctions,
  implications, malformed atoms, undeclared predicates, and wrong arities with
  precise diagnostics.
- Negative guards such as `not done` are valid DFA structure and must not
  compile to synthetic negative atomic subgoals. A signed negative leaf may
  call either a certified positive-sibling branch or a single PDDL deleter only
  when symbolic execution proves an exact net `MustDelete` for the forbidden
  atom, preservation of all positive siblings, and no forbidden completion
  `MayAdd`. If no such branch exists, the literal remains observation-only.
- Every positive repair in a transition containing negative predicate guards
  must carry a completion-level conditional `MayAdd` preservation certificate.
  If an unfiltered atomic module may add a forbidden atom, enforce a query-local
  action-only branch selection that preserves positive siblings and all negative
  guards; if no non-empty goal-achieving selection remains, reject with
  `negative_guard_not_preserved`. A negative-only edge may use the same
  schema-certified single-action deleter; otherwise it only observes absence.
- On a progress edge entering an accepting state, a query-local negative helper
  may place a schema-certified action that satisfies the complete signed guard
  before partial deleters. Retain every partial deleter as an applicability
  fallback. Do not apply this preference to non-accepting edges without a
  separate future-obligation preservation certificate, and never recognize a
  domain, predicate, or action name.
- Mixed Boolean/numeric conjunctions use complete action-only net Boolean
  effects and constant-integer numeric deltas. Helpers are indexed by the full
  grounded/lifted literal, not only by predicate name. A literal without a
  complete preserving branch is observation-only. Negated numeric equality is
  monitored exactly but has no invented numeric-disequality achievement action.
- Literals common to every waiting self-loop cube of an Until source state are
  source invariants. In the supported benchmark fragment, such a state has one
  positive progress literal. Query-local action-only branches must preserve the
  source invariants at every primitive prefix until that progress literal is
  established. Predicate preparation decreases missing producer preconditions;
  numeric preparation decreases a constant-bounded prerequisite deficit while
  leaving the target numeric fluent unchanged.
- Repeated numeric progress that could consume a protected object must use a
  schema-derived non-unification guard. An unavoidable consuming step is allowed
  only at the exact target predecessor, and a query-local observed-equality base
  case terminates recursion. All nested support variables must be alpha-renamed
  away from query variables and outer-producer variables before composition.
  Positive numeric equality may use a query-local PDDL action only when a
  constant integer effect proves strict unit progress, or when a non-unit
  effect is enabled at the exact predecessor value. If a single action proves
  every Boolean and numeric obligation of a guard, that complete net-effect
  certificate may discharge an otherwise cyclic mixed serialization.
- Accepting self-loops labelled `true` are allowed as DFA plumbing and should
  not compile to atomic subgoals.
- Jason's PDDL environment runs a query-local deterministic DFA monitor after
  the initial valuation and after every successful primitive action. The
  monitor exposes only query-local state and acceptance beliefs required by the
  appended controller; these are not PDDL fluents, atomic modules, or exported
  actions. This primitive-step boundary supports the benchmark's strong-until
  semantics and detects intermediate violations.
- Every progress transition uses one query-local `trans` controller enabled by
  a zero-arity query entry proposition. Its certified literal order is compiled
  into a balanced binary repair tree. Internal tree nodes only dispatch to two
  child ranges; each leaf either observes one satisfied literal or calls its
  atomic module once. A separate `trans_done` helper retries only while the
  runtime monitor remains in that transition's source state. Leaving the source
  state completes the helper, including when one atomic macro crosses more than
  one DFA edge; the top-level dispatcher then follows the monitor's actual
  state. The tree is query-local control structure, not a domain fluent or a
  second temporal fast path.
- The balanced tree bounds sibling-plan fan-out by two, visits all positive
  literals in linear controller work per pass, bounds each generated plan body
  by two steps, and has logarithmic nesting depth. A direct linear realization
  would preserve the same certified order but place all leaf calls in one body,
  making its plan representation and intention continuation grow with guard
  size. The tree does not reduce primitive PDDL action count or choose the
  literal order; threat and preservation certificates determine the order first.
  A singleton transition is the identity case: one leaf calls one atomic module
  and the done helper rechecks the same transition.

```asl
query.

+!g_query : query & g_query_monitor_accepting <- true.
+!g_query : query & g_query_monitor_state_q0 <-
	!g_query_trans_1;
	!g_query.
+!g_query_trans_1 : query <-
	!g_query_trans_1_repair_1_1;
	!g_query_trans_1_done.
+!g_query_trans_1_repair_1_1 : query & on(X,Y) <- true.
+!g_query_trans_1_repair_1_1 : query & not on(X,Y) <- !on(X,Y).
+!g_query_trans_1_done : query & not g_query_monitor_state_q0 <- true.
+!g_query_trans_1_done : query <- !g_query_trans_1.
```

- A conjunctive transition is serialized only from complete conservative
  may-delete summaries of the final selected atomic modules. Reject incomplete
  summaries, cyclic threat graphs, and uncertified numeric conjunctions; never
  fall back to parser order or monotonic step helpers.
- Every distance-reducing DFA edge receives a query-local dispatch plan guarded
  by the current runtime monitor state. Same-source/same-target MONA cubes are
  grouped only by their common achievement objective; the runtime monitor still
  evaluates the original complete cubes. This supports state-dependent DFA
  execution without restoring legacy domain-level `tg_state` beliefs.
- Runtime monitoring is semantically exact for the declared formula fragment;
  action strategy synthesis is not complete for arbitrary PDDL-times-LTLf
  products. A valid query may still fail or time out when no certified atomic or
  schema action can establish a required progress objective.

## Temporal Goal Validation

- Model predictions use the exact eight-key lifted LTLf payload defined by
  `src/temporal_specification/prediction_validation.py`. Validation is
  fail-closed over schema, atom-table closure, PDDL catalogue membership,
  arity, parameter type, numeric equality, and the declared operator fragment.
- Translation correctness is semantic rather than textual. Gold and predicted
  atoms are canonicalized by PDDL semantic identity, compiled by the real
  LTLf2DFA/MONA path, and checked for exact finite-trace language equivalence by
  reachable product-automaton exploration. Persist a distinguishing valuation
  trace when equivalence fails.
- The sealed construction witness is a separate consistency check. Replay its
  primitive PDDL actions, compare state fingerprints, ground with the hidden
  assignment, and require both gold and predicted DFAs to accept. Witness
  acceptance must not replace DFA language equivalence.
- For generated execution traces, VAL checks action legality against a generated
  copy of the PDDL problem whose original goal is replaced only by
  `(:goal (and))`. Temporal success is decided independently by gold-DFA
  acceptance over the complete replayed state trace. Do not use the original
  PDDL achievement goal as a TEG success criterion.
- Keep translation errors, validation infrastructure failures, benchmark-audit
  inconsistencies, VAL failures, and DFA rejection as distinct report statuses.

## Selected Benchmark Scope

The domain groups are evaluation coverage, not backend-routing classes. The
current scope includes every MOOSE companion benchmark domain that has a direct
non-empty `training/` and `testing/` split, plus project-added
feature-definable serialized-width benchmarks.

| Group | Domains |
| --- | --- |
| ESHO classical domains | `barman`, `ferry`, `gripper`, `logistics`, `miconic`, `rovers`, `satellite`, `transport` |
| Numeric fluent domains | `numeric-ferry`, `numeric-miconic`, `numeric-minecraft`, `numeric-transport` |
| Feature-definable serialized-width domains | `blocksworld-clear`, `blocksworld-on`, `blocksworld-tower`, `depots` |

Numeric MOOSE domains are included for MOOSE-faithful benchmark coverage and
for the explicitly supported bounded-integer compiler fragment. This fragment
has executable Jason belief updates, constant-integer numeric effects,
schema-certified equality progress, primitive-step DFA monitoring, neutral-goal
VAL validation, and independent DFA trace acceptance. Do not generalize this
claim to arbitrary arithmetic expressions, continuous values, or unrestricted
numeric planning.

## Hard Constraints

- PDDL only. Do not reintroduce HDDL or HTN code.
- Do not emit synthetic achievement names such as `achieve_*`, `transition_*`,
  or `dfa_state` in final ASL.
- `obj_tp/2` is the only reserved non-PDDL context predicate allowed in final
  ASL; it represents PDDL object type membership and must not appear in plan
  bodies or exported PDDL action traces.
- Query-specific names such as `g_query_17`, `g_query_17_trans_1`,
  `g_query_17_trans_1_repair_1_8`, and `g_query_17_trans_1_done` are allowed
  only as temporal wrapper goals. They must not be used as atomic domain
  modules, world-state fluents, or exported PDDL actions.
- Use stored or provided LTLf artifacts unless explicitly asked to regenerate
  them with a language model.
- Keep external generalized-planning code under `.external/`; do not vendor
  paper implementations into `src/`.
- Never run external generalized-planning learners without a hard memory guard.
  Keep the default limit at or below `16GiB`.

## Benchmark Data

Formal data should live under:

```text
src/domains/<domain>/domain.pddl
src/domains/<domain>/train/*.pddl
src/domains/<domain>/test/*.pddl
src/domains/<domain>/source.json
```

Use deterministic splits:

```text
MOOSE official domains = source training/ as train and testing/ as test
blocksworld-clear and blocksworld-on = KR 2025 learner-policies official
no-constants train/test folders
blocksworld-tower = floor(1/4 * instance_count) train and remaining instances
as test
depots = floor(1/2 * instance_count) train and remaining instances as test,
because the D2L source has only 22 instances and the compiler needs broader
stacking/transport evidence
```

The current selected corpus is materialized from pinned sources through
`scripts/materialize_achievement_benchmarks.py`. Do not add old generated
12-family routing datasets back into `src/domains` unless the formal scope
changes.

## External Backend Notes

Pinned backend code remains under `.external/gp-backends/` plus `.external/moose`.

Useful audit commands:

```bash
bash scripts/setup_mona.sh
uv run python scripts/gp_backend_audit.py status
uv run python scripts/gp_backend_audit.py usage
uv run python scripts/gp_backend_audit.py capability
uv run python scripts/gp_backend_audit.py moose-atomic-command --domain-file src/domains/ferry/domain.pddl --training-dir src/domains/ferry/train --save-file tmp/moose-atomic/ferry.model --timeout-seconds 1800
uv run python scripts/gp_backend_audit.py moose-readable-summary --policy-file .external/moose/exact-runs/ferry-seed0.model.readable --domain-name ferry
uv run python scripts/gp_backend_audit.py moose-readable-compile-asl --policy-file .external/moose/exact-runs/ferry-seed0.model.readable --domain-name ferry --output-dir tmp/moose-atomic/ferry-library
uv run python scripts/gp_backend_audit.py moose-readable-compile-asl --policy-file tmp/moose-blocks-e2e/blocks-probe-first4.model.readable --domain-file src/domains/blocksworld-tower/domain.pddl --domain-name blocksworld-tower --validated-policy-lifting --output-dir snapshots/moose_blocks_minimal_modules
uv run python src/main.py compile-moose-atomic-library --policy-file tmp/moose-blocks-e2e/blocks-probe-first4.model.readable --domain-file src/domains/blocksworld-tower/domain.pddl --domain-name blocksworld-tower --validated-policy-lifting
uv run python src/main.py append-lifted-temporal-goal --domain-file src/domains/blocksworld-tower/domain.pddl --ltlf-goal-json artifacts/input/blocksworld_lifted_ltlf.json --query-id query_1
```

MOOSE local reproduction notes are historical evidence that Ferry can be
reproduced with the official artifact. They do not prove that every selected
domain has a final ASL compiler path.

## Workflow

- Use `uv` for Python commands.
- Use test-driven changes for behavior changes.
- Run relevant tests after changing generation, parsing, validation, benchmark
  data, or AgentSpeak rendering.
- Batch experiment scripts must not stay silent for long-running domain/test
  loops. Follow the timestamped batch shell-script style: after each domain or
  test instance finishes, print one concise terminal progress line with the
  domain name, test identifier when applicable, status, and elapsed time or
  artifact path.
- Use `TO-DO-LIST.md` as the active progress tracker.
- Do not create extra documentation files unless explicitly requested.
- Do not delete files with recovery value without a clear git-backed reason.
- When explaining architecture with dense research terms, define every term in
  place and give a concrete example. This is mandatory for input/output lists,
  pipeline diagrams, and module names. Do not only write
  `MOOSE readable policy -> PDDL domain -> training problems -> CompactRecursiveModuleProgram`.
  Instead, explain each item inline: a MOOSE readable policy is the
  `policy --dump-policy` first-order decision-list artifact, for example a rule
  whose goal condition is `(on block0 block1)` and whose action sequence is
  `(pick-up block0) (stack block0 block1)`; a PDDL domain is the predicate and
  action schema file, for example an action `stack(?x, ?y)` with preconditions
  `holding(?x)` and `clear(?y)`; training problems are the PDDL instances used
  by the backend as evidence, for example small Blocks problems containing
  singleton `on(a,b)` goals; a CompactRecursiveModuleProgram is the selected
  recursive atomic module set before AgentSpeak(L) rendering, for example
  `+!on(X,Y)` calling `!clear(Y)` before `stack(X,Y)`.
