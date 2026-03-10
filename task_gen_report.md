# EmToM: Automated Generation of Epistemic Theory-of-Mind Tasks for Multi-Agent Evaluation

> **TL;DR:** An LLM agent loads a 3D household scene, then iteratively drafts a multi-agent task requiring Theory of Mind reasoning—expressed as a formal goal (PDDL), private per-agent secrets, and interaction constraints (e.g., communication limits, room restrictions). Each candidate passes three validation gates: structural checks (PDDL solvability + trajectory replay in the Habitat simulator), a dual-model judge council (Kimi K2.5 + GPT-5.2, consensus required), and a full LLM-agent benchmark.

## 1 Overview

We describe the EmToM task generation pipeline, a semi-automated system for producing multi-agent collaboration challenges that require Theory of Mind (ToM) reasoning in embodied environments. Tasks are expressed as PDDL goal specifications grounded in 3D household scenes from the Habitat simulator—a physics-based platform for embodied AI in which agents navigate, manipulate objects, and communicate within reconstructed indoor environments from the Habitat Synthetic Scenes Dataset (HSSD). Each generated task defines a scenario in which two or more agents must coordinate—cooperatively, competitively, or in a mixed-motive setting—while reasoning about each other's beliefs, observations, and private information.

The pipeline proceeds in four stages. **(1)** A ReAct-style LLM agent designs tasks through iterative scene exploration and self-correction. **(2)** A multi-model council judge scores candidate tasks on structural and narrative criteria. **(3)** A simulation-in-the-loop verification system validates task solvability through deterministic trajectory execution. **(4)** An evolution pipeline escalates task difficulty through calibrated benchmarking against a model ladder. Figure 1 provides a schematic overview of the full data flow.

```
                           ┌──────────────────────────────────┐
                           │     Template / Seed Task         │
                           └───────────────┬──────────────────┘
                                           ▼
                           ┌──────────────────────────────────┐
                           │   new_scene[N]: Load HSSD scene  │
                           │   (rooms, furniture, objects,     │
                           │    agent spawn positions)         │
                           └───────────────┬──────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────────┐
                    │            ReAct Agent Loop                     │
                    │                                                 │
                    │  bash[...]  ──────► edit working_task.json      │
                    │       │                                         │
                    │       ▼                                         │
                    │  verify_pddl[] ──► syntax + solvability + ToM  │
                    │       │                                         │
                    │       ▼                                         │
                    │  judge[] ─────────► multi-model council         │
                    │       │             (Kimi K2.5 + GPT-5.2,       │
                    │       │              8–10 criteria,             │
                    │       │              both must agree)           │
                    │       ▼                                         │
                    │  verify_trajectory[] ► planner + simulator     │
                    │       │                                         │
                    │       ▼                                         │
                    │  test_task[] ─────► LLM-agent benchmark        │
                    │       │                                         │
                    │       ▼                                         │
                    │  submit_task[] ───► final task + ToM metadata   │
                    └──────────┬──────────────────────────────────────┘
                               ▼
                    ┌──────────────────────────────────────┐
                    │  Output: data/emtom/tasks/<id>.json  │
                    └──────────┬───────────────────────────┘
                               ▼
               ┌──────────────────────────────────────────────┐
               │          Evolution Pipeline                   │
               │                                               │
               │  Seed: copy existing + generate easy tasks    │
               │       │                                       │
               │       ▼                                       │
               │  For each model in ladder:                    │
               │    1. Benchmark pool against model            │
               │    2. Compute pass rate ρ                     │
               │    3. ICL sampler: select examples            │
               │    4. Generate harder tasks                   │
               │    5. Repeat until ρ ≤ ρ*                     │
               └───────────────────────────────────────────────┘
```
**Figure 1.** End-to-end data flow of the EmToM task generation pipeline. The inner loop (ReAct agent) iterates over tool calls until a task passes all validation gates. The outer loop (evolution pipeline) escalates difficulty across a model ladder.

## 2 Task Representation

We represent each generated task as a structured JSON object (`GeneratedTask`) that fully specifies the scenario, agent configurations, goal conditions, and interaction constraints. The representation is designed to be self-contained: a single task file encodes everything needed to instantiate, execute, and evaluate an episode in the Habitat simulator.

The **task description** is a natural-language string written without agent-specific information or object IDs (e.g., *"Two roommates must coordinate to prepare dinner, but each knows about different ingredients hidden throughout the house"*). Each agent receives **private secrets**—natural-language strings that create information asymmetry and motivate communication (e.g., *"You know the radio is hidden inside the locked cabinet in the bedroom"*). Agents are also assigned a **permitted action set** drawn from the environment API: `{Navigate, Open, Close, Pick, Place, UseItem, FindObjectTool, FindReceptacleTool, FindRoomTool, Communicate, Wait}`.

The **goal specification** is encoded as an inline PDDL problem string (`problem_pddl`), which serves as the sole canonical goal format. The goal formula may use standard logical connectives (`and`, `or`, `not`) over world-state predicates (`is_on_top`, `is_inside`, `is_in_room`, `is_open`, `is_closed`, `is_held_by`, among others), as well as the epistemic operator $K$ for knowledge goals at nesting depths 0–3. A goal such as $K(\text{agent\_1}, \text{is\_on\_top}(\text{cup}, \text{table}))$ requires agent\_1 to *know* that the cup is on the table—a condition satisfied only when the agent has directly observed the predicate or received the information via communication (evaluated by the belief state tracker described in Section 7.2).

**Mechanic bindings** impose structured constraints on agent interactions. The system supports nine mechanic types, of which six are commonly used in ToM task design: *room restriction* confines agents to designated rooms, preventing direct observation of distant state; *limited bandwidth* caps each agent's total Communicate actions, forcing efficient information transfer; *restricted communication* limits which agents may send messages to which recipients; *unreliable communication* introduces a failure probability on message delivery; *remote control* links trigger-target object pairs so that one agent's action on a trigger object affects a target object elsewhere; and *conditional unlock* gates container access on prerequisite conditions. Three additional mechanics—*inverse state*, *state mirroring*, and *irreversible action*—provide further expressive range for non-epistemic challenges. Each mechanic binding specifies the affected agents, objects, rooms, and parameters (e.g., message limits, failure probabilities, allowed targets).

Each task is assigned a **category**—cooperative, competitive, or mixed—which determines the evaluation semantics (Section 7). Finally, the task carries **ToM metadata**: a `tom_level` (1–3) indicating the minimum depth of epistemic reasoning required, and a `tom_reasoning` string explaining the epistemic structure, both computed automatically from the PDDL goal.

## 3 Task Generation Agent

### 3.1 Architecture and Workflow

The generator is a ReAct-style LLM agent (typically GPT-5.2 or Claude Opus) that iteratively designs tasks through a cycle of reasoning, tool use, and self-correction. At each step, the agent produces a natural-language `Thought` analyzing its progress, followed by an `Action` selected from a fixed tool set (Section 3.2). The loop terminates when the agent submits a task that passes all validation gates, or when the iteration budget is exhausted.

The agent's system prompt is a structured document assembled at runtime from several components: response format instructions, tool descriptions, a prescribed 9-step workflow, category-specific design rules, the complete registry of available PDDL predicates and scene items, mechanic descriptions, documented design pitfalls (e.g., non-articulated furniture cannot be opened, mechanics must be necessary for goal achievement, sparse scenes produce trivial tasks), epistemic goal ($K$) guidelines with worked examples at depths 0–2, and a diversity section listing structural patterns of previously generated tasks. Template placeholders are resolved with live data from the simulator environment and object/mechanic registries.

The prescribed workflow proceeds as follows: **(1)** load a scene with `new_scene`, **(2)** inspect example tasks for reference, **(3)** design and edit the task JSON, **(4)** verify PDDL correctness with `verify_pddl`, **(5)** submit for judge evaluation, **(6)** iterate on judge feedback, **(7)** verify the golden trajectory in the simulator, **(8)** run a full LLM-agent benchmark with `test_task`, and **(9)** submit the validated task. The ordering enforces that structural validity precedes qualitative review, which in turn precedes empirical testing.

### 3.2 Tool Set

The agent has access to eight tools that span environment interaction, validation, evaluation, and task lifecycle management.

**Environment interaction.** `new_scene[N]` loads a random HSSD scene with $N$ agents via a subprocess (required for a fresh OpenGL context). We validate that each scene contains at least 5 locatable objects (4 for competitive tasks) distributed across $\geq 2$ rooms; scenes failing this check are rejected, with up to 5 automatic retries. A `keep` variant preserves the current task edits while swapping the scene. `bash[cmd]` provides sandboxed shell execution restricted to a whitelist of commands (including `cat`, `grep`, `jq`, `python3`, `sed`, `awk`, `find`, `rm`, `mv`, `cp`, and shell control structures) and confined to paths within the working directory.

**Validation.** `verify_pddl[]` checks PDDL syntax, compiles the goal against the scene's object inventory, analyzes solvability via FastDownward (with fallback to the PDKB solver), and computes the minimum ToM depth. `verify_golden_trajectory[]` regenerates a deterministic trajectory from the PDDL specification using a domain-specific planner, then executes it step-by-step in the Habitat simulator to confirm goal reachability (Section 5).

**Evaluation.** `judge[]` invokes the multi-model council judge (Section 4) as a subprocess, returning per-criterion scores, a pass/fail verdict, and actionable suggestions. `test_task[]` runs a full LLM-agent benchmark (typically 4 episodes) to measure empirical pass rate and collect calibration data; for competitive tasks, it creates cross-model matchups.

**Lifecycle.** `submit_task[]` is gated on all three validation flags (Section 3.3). It generates a canonical task ID from the title, category, scene, and goal hash; computes authoritative ToM metadata via the FastDownward epistemic solver; and writes the final task JSON to the output directory. `fail[reason]` terminates generation entirely, but is gated to require at least 30 iterations (the greater of 30 or 25% of the iteration budget) to have been consumed, preventing premature abandonment.

### 3.3 Validation Gating

The agent maintains three boolean flags: `last_verify_passed`, `last_judge_passed`, and `last_test_passed`. Submission requires the conjunction of all three. Any edit to `working_task.json` via `bash` resets the verify and test flags but preserves the judge flag, since the judge evaluates design quality (which is invariant to minor edits) rather than trajectory correctness. This gating mechanism ensures that every submitted task has been independently validated at three levels: structural (PDDL solvability), qualitative (multi-model judge), and empirical (LLM-agent benchmark).

When the agent exhausts its iteration budget without submitting a task, the generation attempt is recorded as a failure. No partial task is saved. Similarly, if three consecutive judge failures occur, the agent is prompted to load a new scene, as the current scene may lack the spatial structure needed for the target task category.

### 3.4 Seed Tasks

The generator can be initialized from an existing task via the `--seed-task` flag. In this mode, the seed task replaces the blank template as the starting `working_task.json`, and a dedicated section is appended to the user prompt containing the seed task's structure and a directive to produce a novel variant. This mechanism enables targeted generation: given a known-good task, the agent can explore variations on its mechanic combinations, ToM depth, or narrative theme while inheriting a structurally sound starting point. We find that seeded generation produces substantially higher yields for epistemic ($K$-level) tasks compared to unseeded generation.

### 3.5 Context Management

When the conversation history exceeds approximately 80% of the model's context window (estimated at 4 characters per token), older messages between the system prompt and the 10 most recent messages are compressed into a single LLM-generated summary. This preserves key observations and design decisions while reclaiming context for continued iteration. After each task submission, the agent's context is fully reset: the system prompt is refreshed with updated diversity patterns, difficulty guidance and calibration data are re-injected via the `extra_sections` mechanism, and a fresh scene is loaded.

## 4 Multi-Model Council Judge

### 4.1 Architecture

We evaluate task quality using a multi-model council consisting of Claude Opus and GPT-5.2. Both models independently score the candidate task on 8–10 criteria (depending on category and dynamic criteria), and both must agree for the task to pass. We query the models in parallel via `ThreadPoolExecutor`. Requiring dual-model consensus reduces systematic errors that arise from any single model's biases—when the two models disagree on a verdict, we log the disagreement for post-hoc analysis and default to rejection.

### 4.2 Evaluation Criteria

The judge evaluates tasks on seven shared criteria applied to all categories. **Agent necessity** verifies that every agent is indispensable to achieving the goal—removing any single agent should make the task unsolvable. **Secret quality** checks that agent secrets are actionable (they drive behavior), written in natural language, and do not leak cross-agent information. **Task naturalness** ensures that the task description and agent secrets contain no simulator object IDs or PDDL syntax. **Narrative consistency** verifies that the natural-language description faithfully reflects the PDDL goal semantics. **Goal relevance** confirms that every conjunct in the PDDL formula advances the stated objective, with no extraneous predicates. **Mechanic utilization** checks that all listed mechanics are essential to the task design and that removing any mechanic would trivialize the challenge. **PDDL solvability** assesses whether the goal is structurally solvable and, for $K$-level goals, whether the appropriate mechanics (e.g., room restriction, limited bandwidth) are in place to create genuine epistemic challenges.

Each category adds one additional criterion: **task interdependence** (cooperative) requires agents to depend on each other's actions, **goal opposition** (competitive) requires conflicting team objectives, and **subgoal tension** (mixed) requires tension between shared and individual goals. Two dynamic criteria are conditionally added: **user requirements alignment** (when a user query guided generation) and **task novelty** (injected from the diversity tracker, described in Section 6).

### 4.3 Passing Conditions

Let $\mathbf{s} = (s_1, \ldots, s_C)$ denote the vector of criterion scores for a single model, where $C$ is the number of active criteria. A task passes the council if:

$$\forall m \in \{\text{Opus}, \text{GPT-5.2}\}: \quad \bar{s}^{(m)} \geq \tau \quad \wedge \quad \forall i \neq i_{\text{novelty}}: s_i^{(m)} \geq 0.5$$

where $\bar{s}^{(m)}$ is the mean score for model $m$, $\tau$ is the passing threshold (default 0.65; 0.7 in the evolution pipeline), and $i_{\text{novelty}}$ indexes the task novelty criterion. The novelty criterion is treated as *soft*: it contributes to the mean score but is exempt from the per-criterion minimum, preventing structurally sound tasks from being vetoed solely due to pattern similarity with existing tasks.

### 4.4 Difficulty-Calibrated Rubrics

The judge supports three difficulty tiers—easy, medium, and hard—that modulate evaluation rubrics. Easy rubrics accept simpler tasks with 2–3 agents, 0–1 mechanics, and first-order ToM only, appropriate for validating against weaker models. Medium rubrics enforce standard complexity with 1–2 mechanics and ToM levels 1–2. Hard rubrics require 2+ mechanics, tight message budgets, and ToM levels 2–3, targeting frontier models. This calibration ensures that the judge's standards match the intended audience: tasks designed for weaker models are not penalized for simplicity.

### 4.5 Rollout-Grounded Evaluation

When benchmark trajectory data is available from a prior `test_task` invocation, the judge loads the rollout trace—including success/failure outcomes, step counts, percent complete, and per-agent action sequences—and incorporates it into the evaluation prompt. This grounds the qualitative assessment in empirical agent behavior, enabling the judge to identify design issues that are invisible from the task specification alone (e.g., a task that is technically solvable but requires an unreasonable number of coordination steps).

## 5 Simulation-in-the-Loop Verification

We employ three complementary verification mechanisms to ensure that every submitted task is not merely well-designed but physically executable in the Habitat simulator.

**PDDL verification** (`verify_pddl`) performs four sequential checks: syntactic parsing of the `problem_pddl` string, compilation of the goal formula against the scene's object inventory (detecting references to nonexistent objects or unsupported predicates), solvability analysis via the FastDownward planner (with fallback to the PDKB solver), and computation of the minimum ToM depth from the epistemic nesting structure.

**Golden trajectory verification** (`verify_golden_trajectory`) regenerates a deterministic trajectory from the task specification using a domain-specific planner. This planner produces physical actions (Navigate, Open, Close, Pick, Place, UseItem) and, for epistemic goals, derives Communicate steps from the FastDownward epistemic compilation. The resulting trajectory is executed step-by-step in the Habitat simulator via subprocess (each invocation requires a fresh OpenGL context), and the final world state is checked against the PDDL goal. Trajectories are always regenerated from the specification rather than cached, ensuring consistency as the task design evolves during the generation loop.

**Specification validation** (`spec_validator`) applies deterministic structural checks that do not require simulator execution: template placeholder detection (ensuring no `REPLACE_WITH_*` strings remain), mechanic field and schema validation, mechanic binding completeness against scene objects, category-schema consistency, and room-restriction consistency with Navigate targets in the golden trajectory.

## 6 Diversity Tracking

To promote variety across the generated task pool, we maintain a `DiversityTracker` with a persistent log of structural patterns extracted from all submitted tasks. When a new task is proposed, the tracker uses the LLM to extract a 5–15 word structural pattern (e.g., *"Race to find radio hidden in locked container"*) and compares it against up to 30 existing patterns to compute a novelty score $\nu \in [0, 1]$, where 1.0 indicates a completely novel pattern. A word-overlap heuristic serves as a fallback when the LLM call fails.

The novelty score is injected into the judge's evaluation as the `task_novelty` criterion (Section 4.2). Additionally, the diversity section of the generator's system prompt is refreshed after each task submission, surfacing up to 20 existing patterns organized by category. This provides the generator with an up-to-date view of task coverage, steering it toward underrepresented regions of the design space.

## 7 Evaluation

### 7.1 Goal Checking

The `PDDLGoalChecker` evaluates PDDL goal formulas against the simulator's world state by recursively traversing the goal tree. Conjunctions (`and`) require all sub-goals to hold; disjunctions (`or`) require at least one; negations (`not`) invert the inner predicate's truth value. For cooperative and mixed tasks, we use **latched evaluation semantics**: once a conjunct is satisfied, it remains satisfied regardless of subsequent state changes. This is necessary because the Habitat simulator allows agents to physically manipulate objects placed by others—without latching, a cooperative goal could regress when one agent inadvertently moves an object that another agent had already positioned correctly. For competitive tasks, we use **live-state evaluation**: the goal is checked against the current world state at episode termination, meaning opponents can actively undo each other's progress.

Percent Complete (PC $\in [0,1]$) is computed as the fraction of top-level required conjuncts that are satisfied. Success is defined as $S := (\text{PC} = 1)$.

### 7.2 Epistemic Goal Evaluation

Goals containing the epistemic operator $K$ (e.g., $K(\text{agent\_0}, \text{is\_in\_room}(\text{cup}, \text{kitchen}))$) require reasoning about agent beliefs rather than world state alone. We evaluate these goals using a `BeliefStateTracker` that maintains a per-agent belief model updated on three event types: **(a)** room entry, which grants the agent knowledge of all unary and binary predicates involving objects and furniture in the entered room; **(b)** Communicate actions, which transfer the sender's relevant beliefs to the recipient; and **(c)** binary predicate changes (e.g., `is_on_top`, `is_inside`) between objects observed in the agent's current room.

When no belief tracker is available (e.g., during lightweight validation), the evaluator falls back to a conservative check: the inner literal must hold in the world state, which is necessary but not sufficient for the epistemic goal to be satisfied.

## 8 Evolution Pipeline

### 8.1 Curriculum Overview

The evolution pipeline implements an iterative difficulty escalation strategy that produces a task pool calibrated to challenge progressively stronger models. We frame this as a curriculum: the pipeline begins with easy tasks, benchmarks them against a model ladder ordered by capability, and generates harder tasks to drive the pass rate toward a target threshold $\rho^*$.

### 8.2 Seed Phase

The pipeline begins by seeding the task pool from an existing directory of validated tasks. If the pool is smaller than the target seed size (default 30), we generate additional "easy" tasks in parallel—using the task generation agent (Section 3) with easy-difficulty rubrics—and benchmark them against the weakest model in the ladder to establish baseline calibration data.

### 8.3 Tier Loop

The evolution proceeds through one tier per model in the ladder (default: `[gpt-5-mini, sonnet, gpt-5.2]`). At each tier, we execute the following steps:

1. **Benchmark.** We identify tasks missing calibration data for the current model and run benchmarks in parallel.
2. **Assess.** We compute the pool-wide pass rate $\rho$ for the current model.
3. **Generate.** If $\rho > \rho^*$ (default $\rho^* = 20\%$)—indicating the pool is still too easy for the current model—we invoke the ICL sampler to prepare in-context examples and generate harder tasks in parallel batches. After each batch, we re-check $\rho$; generation halts when $\rho \leq \rho^*$ or the generation budget is exhausted.
4. **Advance.** We proceed to the next model in the ladder.

Difficulty is mapped from tier position: seed tasks are "easy," early tiers produce "medium" tasks, and later tiers produce "hard" tasks. Subtask complexity bounds scale accordingly (2–4 for easy, ramping to 5–20 at the highest tiers). The orchestrator also tracks the distribution of ToM levels across the pool and adjusts generation pressure to meet target ratios ($L_1$: 30%, $L_2$: 45%, $L_3$: 25%, with 8% tolerance).

### 8.4 In-Context Learning Sampler

At each generation round, the ICL sampler selects a representative set of benchmark results to provide as few-shot examples for the generator. We select 9 failed tasks and 1 passed task from the pool. Failed tasks are sorted by percent complete in descending order, since partial completions reveal the specific failure modes encountered by the agent (e.g., running out of messages before relaying critical information, or failing to coordinate on object placement). Each example is annotated with benchmark metadata (pass/fail, percent complete, number of steps).

The sampler also constructs a directional guidance string based on the current pass rate $\rho$. When $\rho < 10\%$, the guidance recommends exploring a qualitatively different type of difficulty, since the current difficulty axis may be saturated. When $10\% \leq \rho \leq 30\%$, it recommends amplifying patterns from the failed examples. At higher pass rates, the guidance escalates from combining multiple difficulty sources ($30\text{–}60\%$) to directly exploiting model weaknesses ($60\text{–}95\%$) to generating maximally difficult tasks ($>95\%$).

### 8.5 Parallel Generation

We parallelize task generation across up to 50 concurrent processes, each producing a single task via the generation agent. Categories are assigned round-robin from a configurable list (e.g., `[cooperative, competitive, mixed]`). The orchestrator polls for completed task JSON files, re-evaluates the pool pass rate after each batch, and terminates generation when the target is met. Total process spawns are capped at $5 \times$ the target task count to bound computational cost.

## 9 Production Results

We evaluate the pipeline's yield and the resulting task characteristics across several bulk generation campaigns.

**Yield.** In our largest run (8 GPUs, 3 processes per GPU, 3 tasks per process, approximately 8 hours wall-clock), the pipeline produced 122 validated tasks: 64 cooperative (yield $\approx$ 55%), 30 competitive (yield $\approx$ 25%), and 28 mixed (yield $\approx$ 35%), for an overall yield of 42%. The lower yield for competitive tasks reflects the difficulty of enforcing `agent_necessity`—constructing scenarios where both teams are genuinely indispensable requires careful mechanic design. The three most common judge failure modes were `pddl_solvability` (goal formulas referencing objects unreachable by any valid plan), `mechanic_utilization` (mechanics listed but not required for goal achievement), and `agent_necessity` (tasks solvable by a single agent).

**Epistemic tasks.** Seeded generation (`--seed-task`) reliably produces $K$-level tasks, while unseeded bulk generation has a lower success rate for epistemic goals. Among validated epistemic tasks, $K=1$ tasks achieve 50–100% benchmark pass rates against frontier models. $K=2$ tasks combining `restricted_communication` and `limited_bandwidth` prove extremely challenging: agents allocate a disproportionate fraction of their message budget to coordination overhead rather than task-relevant information transfer, resulting in near-zero pass rates.

**ToM distribution.** The evolution pipeline's ToM-level targeting produces pools with approximately 30% $L_1$, 45% $L_2$, and 25% $L_3$ tasks, within the 8% tolerance specified by the configuration. This distribution ensures coverage across epistemic reasoning depths while concentrating the majority of tasks at the moderate $L_2$ level, which we find provides the best signal for discriminating between model capabilities.
