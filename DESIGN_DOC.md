design doc
Below is a draft design doc you can drop into a repo. It’s written to be operational (definitions
you can implement + mechanics you can build + metrics you can compute).
Design Doc: K-Order Functional Theory of Mind (ToM) in
Multi-Agent Mystery-Room Games
0) Goal and scope
We want to design multi-agent “mystery room” games where task success depends on
Theory of Mind (ToM)—specifically k-order ToM for k \in {0,1,2,3}.
Functional ToM here means: the agent must reason about other agents’
knowledge/beliefs/goals (and nested beliefs) in a way that is causally necessary to achieve
high reward, not merely correlated with good performance.
This doc defines:
1) Core definitions
1.1 Environment model
An episode has:
1. k-order ToM precisely
2. what it means for a task to require k-order ToM
3. design guidelines and mechanics to enforce ToM-necessity
4. simple, measurable success metrics and diagnostics that indicate k-order ToM
World state s \in \mathcal{S}: physical configuration (object locations, door states, triggers,
etc.)
Agents i \in {1,\dots,n}
Private observation histories h_i: what agent i observed + messages received + own
actions
Actions a_i \in \mathcal{A}_i, optional communication m_i
Reward / success R(\tau) as a function of trajectory \tauLet I_i denote information available to agent i (its observation model + restrictions).
1.2 Beliefs and nested beliefs (epistemic ToM)
Define agent i’s belief about the world as:
b_i(s) := P(s \mid h_i).
Define nested beliefs recursively. Informally:
We will use the following shorthand:
This is the psychology-aligned “orders of belief” notion.
1.3 Definition: k-order functional ToM policy
A policy \pi_i is a mapping from history to action:
\pi_i: h_i \mapsto a_i.
We say \pii exhibits k-order functional ToM if it computes actions using an internal
representation that is _behaviorally equivalent to conditioning on nested beliefs of depth up to k,
and this dependence is necessary for high performance on the task family (see §2).
Plain language ladder:
0th-order: beliefs about the world
1st-order: beliefs about another agent’s belief about the world
2nd-order: beliefs about another agent’s belief about your belief about the world, etc.
Order 0 belief: b_i(s)
Order 1 belief: b_i(b_j(s))
Order 2 belief: b_i(b_j(b_i(s))) or b_i(b_j(b_k(s))) depending on the interaction
Order k belief: belief nesting depth k
k=0: track the room/world only (others are just moving objects)
k=1: track what others know/believe (“she didn’t see the key”)
k=2: track what others believe about your knowledge (“he thinks I don’t know the door is
trapped”)1.4 A second, compatible definition (bridge to “level-k reasoning”)
Choose a non-ToM baseline (level-0) for opponents (heuristics: greedy, always-truthful,
always-trust, etc.).
Define recursively:
\pi^{(k)}i \in BR(\pi^{(k-1)}{-i}).
This is useful for constructing baselines and intuitions, but your benchmark should anchor on
the nested-belief definition because “best response depth” can mismatch true belief nesting in
some environments.
2) What does it mean for a task to “require” k-order ToM?
2.1 Necessity (core requirement)
A task family \mathcal{T}_k requires k-order ToM if there exist controlled instances where:
We operationalize “restricted to ≤(k−1)-order” via an information restriction:
Restriction idea: the agent can use:
In practice, you don’t need to prove this formally for each room; you enforce it via design
mechanisms (§3) and validate via diagnostic forks (§4).
2.2 “Success depends on ToM” (practical engineering definition)
k=3: track what others believe about what you believe about them (“she thinks I think she’s
bluffing, so she’ll counter-signal...”)
Any policy restricted to ≤(k−1)-order information (defined below) cannot achieve high
success, but
A policy with access to k-order reasoning can.
world beliefs b_i(s) (order 0),
others’ beliefs about the world (order 1),
... up to order k-1, but not order k representations.A room enforces ToM-dependence when:
3) Design principles to enforce k-order ToM
3.1 The “ToM Necessity Recipe”
To force k-order ToM, include all three:
(A) Nested information structure
Someone has private info; someone else has info about what they know; possibly info about
what they think you know.
(B) Incentives that make messages/actions strategic
Mixed motive helps: agents sometimes benefit from lying, omission, delaying, or misdirection—
but not always, so simple “always distrust” also fails.
(C) Commitment / cost / timing pressure
Make it expensive or impossible to resolve uncertainty by endless verification:
Without (C), agents can probe until they learn the truth, and ToM becomes optional.
1. There is at least one pivotal decision where the correct action differs across conditions
that vary only in an agent mental-state variable (belief/knowledge/intent) while holding the
physical state constant (or observationally equivalent to the player).
2. The agent cannot resolve that decision by:
brute force exploration,
repeated probing / cheap verification,
purely physical cues.
3. Communication is strategic (not guaranteed truthful), or access/visibility asymmetries make
it impossible to directly observe the needed variable.
one-shot levers,
irreversible doors,
limited messages (“one radio token”),
countdown alarms,
resource budgets.3.2 How to build the k ladder (0→1→2→3) with minimal edits
Use the same “room theme,” but adjust what information is private and what is observable.
k=0 tasks (no ToM required)
Design patterns:
k=1 tasks (first-order ToM)
Success requires reasoning about what another agent knows/has seen/can do.
Enforcement mechanisms:
Critical property:
k=2 tasks (second-order ToM)
Success requires reasoning about what another agent believes about your
knowledge/intent.
Enforcement mechanisms:
Critical property:
Fully solvable as single-agent planning under partial observability.
Other agents do not hold unique information needed for the win condition, or their actions
are perfectly predictable and irrelevant.
Hidden keys, deterministic puzzles, static hazards.
Multi-agent is just parallel exploration.
Access/visibility asymmetry: one agent can act, the other can see.
Distributed clues: one agent holds the code, the other controls the safe.
Role abilities: only A can open X, only B can carry Y.
If you ignore others’ mental states and treat them as random, you cannot reliably
coordinate.
Strategic messaging: sender’s optimal message depends on what they think you believe.
Uncertain trust: sometimes the sender is biased, sometimes not.
“Knowledge about knowledge” layers:
A knows X
B knows whether A knows X (or has noisy evidence)
A doesn’t know whether B has that meta-knowledgek=3 tasks (third-order ToM)
Success requires reasoning about what another agent believes about what you believe
about their belief.
Enforcement mechanisms:
Critical property:
4) Mechanics library: concrete room mechanics that create
ToM necessity
Below are building blocks you can plug into rooms.
4.1 Information mechanics (create nested beliefs)
4.2 Incentive mechanics (make communication strategic)
4.3 Commitment mechanics (prevent brute force probing)
The correct action differs depending on whether the other agent thinks you’re naïve vs
skeptical.
Double-bluff / counter-signaling conditions.
Public + private evidence about who knows what.
A strategic equilibrium where the k=2 “obvious” response is exploitable, and the k=3
response avoids exploitation.
There is a predictable k=2 strategy that fails against a strategic opponent who anticipates it.
Asymmetric visibility zones: A can see inside the kitchen, B cannot.
Private clue tokens: only one agent receives a note with code/target.
Meta-clues (“clues about clues”): B gets a note: “A’s clue is outdated 50% of the time,” or
“A probably saw the trap sign.”
Noisy sensors: one agent has a reliability indicator unknown to others.
Observational ambiguity: two physical states look identical to A but not to B.
Hidden personal bonus/penalty tied to which branch is chosen.
Role goals that conflict at the margin (mixed motive): shared win + private preference.
Reputation cost: lying can be punished if detected.
Scoring tradeoffs: speed vs safety, team reward vs personal reward.4.4 “Pivotal fork” mechanic (the diagnostic backbone)
Design every ToM room around at least one fork:
Example forks:
5) What does “success” mean? Metrics that are simple,
measurable, verifiable
You want two layers:
5.1 Task success metrics (primary)
Pick one as your headline metric:
These are easy but not ToM-specific.
5.2 ToM-specific diagnostic metrics (the key)
One-shot commit: choose door A or B, then the other locks.
Limited messages: one message each, or one shared “radio token.”
Resource budget: opening cabinets costs time points; too many opens triggers alarm.
Irreversible actions: pulling lever closes a door permanently.
Time pressure: countdown; every verification step consumes scarce time.
Two actions A and B are both plausible given physical evidence.
Optimal choice depends on an unobserved mental-state variable.
Trust partner’s claim vs verify yourself
Commit now vs wait
Share clue vs conceal
Bluff vs truth-tell
1. Task success (did they win?)
2. ToM diagnostic success (did they take the action that only makes sense under k-order
reasoning?)
Binary win rate: \mathbb{1}[\text{goal achieved}]
Time-to-success: steps or seconds to win (lower is better)
Score: weighted combination of shared goal + constraint violations + timeCreate diagnostic decision points (2–4 per episode), each tagged with a ground-truth
“correct” choice under different mental-state conditions.
Metric A: Counterfactual sensitivity to nested-belief variables
You generate pairs of episodes where:
Then measure whether the agent’s action distribution changes appropriately.
Operational metric:
\Delta_k = \Pr[a = A \mid \text{condition 1}] - \Pr[a = A \mid \text{condition 2}],
where conditions differ only in k-order belief variables.
If \Delta_k \approx 0, agent is not using that order of ToM.
Metric B: “Branch correctness” on ToM forks
For each fork, define which action is optimal under each latent mental-state assignment.
Score:
This is very interpretable.
Metric C: Message pragmatics score (when communication exists)
Compute whether messages are tailored to the receiver’s knowledge state.
Examples:
Simple metric:
physical layout is the same,
the agent’s own observations are the same,
only others’ private observations or knowledge-about-knowledge differs.
Fork accuracy: percent of forks where the agent chose the action consistent with the
correct nested-belief model.
Does agent omit info the receiver already knows?
Does agent include info the receiver lacks?
Does agent choose different phrasing/strategy when receiver is skeptical vs trusting?
Informational gain: does message reduce receiver uncertainty about the key latent
variable?Metric D: Costly verification tradeoff curve
In ToM-required rooms, a key signature is choosing when to verify.
Measure:
A k=0/1 heuristic typically cannot condition verification on partner belief models.
6) How to claim “this room tests k=2” (minimal checklist)
A room is a credible k=2 test if:
For k=3, same but with an additional manipulation that changes what the other agent believes
about what you believe about them.
7) Concrete example templates (plug-and-play)
Template 1: “One-shot safe selection” (k=1→k=3 scalable)
Strategy shift: does message change when the receiver’s belief-about-sender changes?
rate of verification actions as verification cost increases
whether verification is used selectively when partner is likely to lie
1. There exists a pivotal fork where the optimal choice depends on b_i(b_j(s)) and on
b_i(b_j(b_i(s))) (what they think you know).
2. The agent cannot disambiguate the fork by direct observation without paying a large cost or
losing.
3. The other agent’s incentives make their action/message contingent on their model of you
(naïve vs skeptical).
4. You have at least one controlled manipulation that flips only the k=2 latent variable while
holding the agent’s local observation fixed.
Two safes: only one contains the objective.
A can open safes; B can inspect labels but cannot open.
k=1: B sees which safe is correct; A must trust B.
k=2: B sometimes benefits from misleading; A knows B might be biased; B doesn’t know
whether A knows.
k=3: B knows that A knows B might be biased, and may counter-signal.Fork: open safe 1 vs safe 2.
Template 2: “Linked-cabinet / remote trigger” (your current
mechanic; strong)
Fork: commit to opening cabinet X now vs delay / choose Y.
Template 3: “Public audit with strategic reports”
Fork: trust report vs verify.
8) Implementation notes (how to encode ToM ground truth)
To compute ToM metrics, log ground truth latent variables:
Then you can label each fork with a “correct action” under each latent assignment.
9) Summary: what you will be able to claim
With this framework, you can say:
Action in one room triggers state change elsewhere.
Visibility and access are split across agents.
Add: limited messages + mixed motive + meta-clues.
Agents must set a configuration (open/closed pattern).
Success depends on passing an audit that uses reports; inconsistent reports trigger
penalty.
One agent has incentive to misreport if they think others can’t verify.
K_i: what agent i knows (clue IDs received, observations seen)
M_i: agent i’s motive type (honest, biased-to-A, biased-to-B)
E_{ij}: whether i knows that j knows a specific fact (meta-knowledge)
Message budgets, action costs, timers
“Our k=1 rooms require reasoning about other agents’ knowledge/access.”
“Our k=2 rooms require reasoning about other agents’ beliefs about the player’s knowledge
(strategic signaling).”
“Our k=3 rooms require third-order belief reasoning (double-bluff/counter-signaling).”“We measure ToM not only via win rate but via diagnostic fork accuracy and sensitivity to
controlled nested-belief manipulations.”