## EmToM-B Discussion Notes

### Level-K Reasoning Framework

The cognitive hierarchy / Level-K thinking framework maps naturally onto the ToM levels needed in EmToM-B:

- **L0 (No ToM):** Broadcast everything, act greedily. This is the "trivial strategy" the paper flags as a current blocker. The agent treats communication as a dump of all observations with no modeling of what the other agent knows or will do.
- **L1 (First-order ToM):** Selective broadcasting. The agent models what the other agent knows and only communicates what's missing or decision-relevant. Requires tracking partner's observation history.
- **L2+ (Higher-order ToM):** "I think you think I know X." Enables deception, bluffing, and strategic withholding. Relevant for competitive/mixed-motive tasks where opponent modeling matters.

Key insight: the level of reasoning required depends on the task structure. In purely cooperative tasks, L1 may suffice. In competitive or mixed-motive tasks, L2+ becomes necessary because opponents are also reasoning about you.

### Mechanisms That Force Higher-Order ToM

**Constrained communication bandwidth.** If agents can only send N tokens per round, they must prioritize what to communicate. This naturally pushes beyond L0 (can't broadcast everything) and into L1+ (must model what the receiver needs most). Tighter bandwidth forces higher-order reasoning about what the receiver already knows vs. what would change their plan.

**Irreversible actions and information.** When actions can't be undone (e.g., a door once locked stays locked, a resource once consumed is gone), the stakes of miscoordination increase. Agents must reason about the consequences of their actions on the partner's belief state and future options _before_ acting. Irreversibility also makes deception more costly/powerful in competitive settings.

**Proximity-based communication (#idea2).** If agents can only talk to others within physical proximity, this creates natural information asymmetries that evolve over time. It also introduces a strategic dimension to movement: do you go toward the objective, or toward your teammate to share information first?

**Overheard strategies / environment artifacts.** If communication between teammates can be intercepted by the opposing team, this forces higher-order reasoning. You can no longer plan openly; you must reason about what the opponent heard, what they'll infer, and whether to use that channel for deception. This directly pushes tasks into L2+ territory.

### Ideas for Benchmark Design

**#idea1: Redundant/noisy secrets in agent context.** Pad each agent's private information with irrelevant or misleading "secrets." This tests whether agents can identify what's actually decision-relevant vs. noise. An L0 agent broadcasting everything now actively hurts performance because it floods the partner with irrelevant info (or leaks useful info to opponents). This also gives a natural metric for communication quality: how much of what was communicated was actually used in the partner's decision-making?

**#idea2: Proximity-based communication.** (See above.) Creates dynamic, spatially-grounded information asymmetries.

### Evaluation Challenges

**Intransitivity in competitive settings.** For zero-sum or mixed-motive tasks, agent A beating agent B and B beating C does not imply A beats C. This is the intransitivity phenomenon. Standard Elo-style rankings break down. The right evaluation tool here is the **Nash averaging** approach (e.g., from Balduzzi et al.), which finds the mixture of strategies that is unexploitable and uses that to rank agents.

**"Better" agents can fail on "easier" tasks.** A more capable agent (higher Level-K) might overthink a task that a simpler agent solves by brute force. For example, an L2 agent might model an opponent as strategic when the opponent is actually L0, leading to suboptimal play. This is a known issue with cognitive hierarchies and suggests evaluation should report performance _per level of task complexity_, not just aggregate scores.

**Measuring ToM necessity.** Open question: given a task, how do you formally verify that it _requires_ Level-K reasoning? One approach is to show that an L(K-1) agent provably cannot achieve optimal performance on the task, but an L(K) agent can. The paper currently uses an LLM judge for this, which is brittle.

### Two Research Directions

1. **Higher-order cognitive hierarchies.** Designing tasks and mechanics that provably require L2+ reasoning. The Level-K framework gives a formal language for this. The key contribution would be a taxonomy of tasks by the minimum Level-K needed, plus empirical evidence that current models plateau at a specific level.
2. **Evolving environments/tasks.** Open-ended task generation that adapts to the capabilities of the agent population. As agents get better, the benchmark generates harder tasks (more mechanics composed, tighter constraints, higher ToM requirements). This addresses the benchmark saturation problem and connects to ideas from open-ended learning and co-evolution.