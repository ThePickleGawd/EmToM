# PARTNR PDDL Task Spec (Single-Format)

Status: Draft v1  
Scope: Replace all goal/subtask legacy formats with one formal PDDL problem specification.

## 1. Objective

Tasks MUST define formal goals through exactly one inline PDDL problem field.
There MUST be no parallel goal representation (`goals`, `pddl_goal`, `pddl_ordering`, `pddl_owners`, `subtasks`, `success_condition`) in task payloads.

This spec aims to provide formal guarantees over a clearly defined symbolic abstraction of PARTNR.

## 2. Single Source Of Truth

Each task payload contains one authoritative field:

1. `task.json.problem_pddl` (required, authoritative)
2. `task.json.pddl_domain` (required, must match `problem_pddl` `:domain`)

`problem_pddl` is the only normative source for:

1. `:objects`
2. `:init`
3. `:goal`
4. optional `:metric`

`task.json` MUST NOT contain any other goal logic fields. It may contain:

1. task metadata (`task_id`, `title`, `scene_id`, `episode_id`, `num_agents`)
2. agent-facing natural language (`task`, `agent_secrets`, etc.)
3. mechanic declarations that compile into symbolic predicates

## 3. Domain Contract

`problem_pddl` MUST target one versioned domain, e.g. `emtom`.

Shared domain file must exist on disk at:

1. `emtom/pddl/domains/<pddl_domain>/domain.pddl`

Domain versioning rules:

1. Domain is immutable once released.
2. Any predicate/action semantic change requires a new domain version.
3. Task files pin exact domain name and version.

## 4. Formal Semantics Boundary

Formal guarantees apply to the symbolic model, not raw simulator state.  
Therefore, this spec defines explicit mapping functions:

1. State abstraction `alpha`: PARTNR state -> symbolic state
2. Action abstraction `beta`: PARTNR action event -> symbolic grounded action
3. Goal interpretation `G`: PDDL goal formula truth over symbolic state/belief model

### 4.1 State Abstraction (`alpha`)

`alpha` MUST be deterministic and total over supported predicates.

For every supported predicate `p`, define:

1. source of truth in simulator/game state
2. extraction rule
3. update cadence (per step, end-state only, etc.)
4. known uncertainty mode (none/ambiguous/stochastic)

No predicate may exist in domain without an extractor spec.

### 4.2 Action Semantics (`beta`)

For each supported symbolic action schema, define:

1. grounding from real action arguments
2. symbolic precondition check
3. symbolic effect set
4. observability/epistemic update rule

No symbolic effect may be "implicit". All effects must be explicit in domain semantics.

### 4.3 Goal Semantics (`G`)

Goals are evaluated against symbolic state and epistemic model only.
If using `K`/`B`, evaluator MUST use belief-tracker semantics, not inner-literal fallback.

## 5. Allowed PDDL Profile

To keep semantics clean and verifiable, v1 profile is restricted:

1. Ground object constants only in goals (no free variables).
2. Goal operators: `and`, `not`, `K`, `B` (optional `or` only if planner supports certified branch semantics).
3. No quantifiers in goals in v1.
4. Predicate vocabulary must be a strict subset of domain predicates.
5. All object symbols in goal must appear in `:objects`.

If a construct is not supported end-to-end by parser + planner + checker + evaluator, it is forbidden.

## 6. No-Hack Rules

The runtime MUST satisfy all:

1. No legacy goal ingestion path for tasks that provide `problem_pddl`.
2. No auto-conversion from legacy task fields.
3. No unwrapping epistemic goals into plain literals in certified evaluation.
4. No hidden "fallback success" based on non-PDDL fields.
5. No dual ownership semantics outside PDDL (ownership must be encoded in problem/domain semantics if needed).

If any fallback is still required for operations, it must run in explicit `legacy_mode`, never in default mode.

## 7. Validation And Guarantees

Every submitted task must pass all gates:

1. Syntax/type gate:
1. parse domain/problem
   2. type-check predicates/arguments
   3. reject unsupported operators
2. Abstraction consistency gate:
   1. compute `alpha(s0)`
   2. compare with `problem.pddl :init`
   3. fail on mismatch
3. Transition conformance gate:
   1. replay reference trajectories
   2. compare predicted symbolic effects vs extracted next symbolic state
   3. require mismatch rate <= configured threshold
4. Goal fidelity gate:
   1. execute trajectory in simulator
   2. evaluate `G(alpha(s_t))`
   3. require agreement with task success label
5. Epistemic fidelity gate (if `K`/`B` used):
   1. verify observability/communication updates
   2. reject tasks where epistemic goals are trivially satisfied unless explicitly allowed

## 8. Why This Still Cannot Be "Perfect"

A fully perfect abstraction (bisimulation) is generally not achievable for PARTNR because:

1. Simulator physics and geometry are continuous; PDDL is discrete.
2. Some action outcomes are effectively stochastic/noisy at the symbolic boundary.
3. Observability and belief are model approximations, not direct ground-truth mental states.

So the correct target is:

1. explicit abstraction boundary
2. measured conformance guarantees
3. strict rejection when abstraction assumptions are violated

## 9. Practical Acceptance Criteria For v1

We can claim "clean formal PDDL support" only when:

1. Task files contain no legacy goal fields.
2. End-to-end evaluation uses only `problem_pddl` for goal truth.
3. Epistemic goals are evaluated via belief tracker (no literal fallback in certified mode).
4. `alpha` extractor spec exists for every domain predicate.
5. CI includes conformance tests and blocks regressions.

## 10. Migration Plan (One-Way)

1. Freeze current mixed format as legacy.
2. Introduce inline `problem_pddl` + strict validator.
3. Remove legacy read paths from generator, verifier, evaluator, submitter.
4. Enable certified mode as default.
5. Delete legacy mode after migration window.

No backward compatibility promises after v1 cutover.
