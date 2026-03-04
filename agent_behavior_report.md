# Evaluation

We evaluate EmToM by benchmarking OpenAI's o3 on 12 tasks stratified by category and ToM level. Each task is run once (single episode). We report aggregate pass rates, analyze action distributions and communication strategies, and ground the failure taxonomy of Section 3 in specific behavioral evidence from agent traces. Throughout, we connect observed behaviors to the level-$k$ framework (Equation 1): a level-0 agent acts without modeling others, a level-1 agent models what others will do, and a level-2 agent models what others believe about what others will do.

## Experimental Setup

We select 12 tasks via stratified sampling: 4 cooperative, 4 competitive, and 4 mixed, covering ToM levels 1 and 2. Cooperative and mixed tasks use 3 agents; competitive tasks use 4 (two teams of two). All agents are powered by o3 via the OpenAI API with a turn budget of 20 (some episodes run to 21 turns due to a final-turn grace period). We record full action histories, inter-agent communication transcripts, and per-turn goal satisfaction.

Cooperative and mixed tasks use *latched* evaluation: once a goal conjunct is satisfied, it remains satisfied regardless of subsequent world-state changes. Competitive tasks use *live-state* evaluation: goals are checked against the world state at episode termination only, so opponent actions can undo previously achieved progress. This distinction is critical for interpreting percent-complete (PC) values: a competitive team may reach 80\% mid-episode but terminate at 0\% if the opposing team reverses their placements.

## Aggregate Results

Table 1 summarizes per-task results. The overall pass rate is 25\% (3/12). All three successes are ToM level 1; all five level-2 tasks fail.

**Table 1.** Per-task results for o3 on 12 EmToM tasks. PC = percent complete at termination. Cat = category.

| Task | Cat | ToM | Pass | PC | Turns | Msgs |
|---|---|---|---|---|---|---|
| Distributed Condition Report | coop | 1 | Y | 100% | 4 | 3 |
| The Quiet Hand-off | coop | 1 | N | 75% | 20 | 7 |
| Tamper-Evident Seal Drill | coop | 2 | N | 80% | 21 | 6 |
| Relay Roll-Call | coop | 2 | N | 75% | 21 | 7 |
| Display Stance Bowl Duel | comp | 1 | N | 0% | 20 | 15 |
| Locker Rally | comp | 1 | N | 80% | 20 | 8 |
| Token Stance Showdown | comp | 1 | N | 71% | 20 | 11 |
| Privacy Mode Tug-of-War | comp | 2 | N | 43% | 21 | 7 |
| Staging Protocol | mix | 1 | Y | 75%* | 8 | 1 |
| Emergency Housekeeping Drill | mix | 1 | Y | 100% | 11 | 3 |
| Quiet-Mode Relay | mix | 2 | N | 83% | 21 | 5 |
| Tri-Room Status Ping | mix | 2 | N | 43% | 21 | 7 |

Successful tasks complete in 7.7 turns on average; all nine failures hit the turn ceiling. (*Staging Protocol passes at 75\% PC because mixed tasks pass when the shared goal is satisfied; the 75\% reflects unmet private agent subgoals.) By category, mixed tasks achieve the highest pass rate (2/4), cooperative tasks pass 1/4, and no competitive task passes. The zero competitive pass rate reflects live-state evaluation: on *Display Stance Bowl Duel*, both teams achieve their respective physical subgoals mid-episode, but each team reverses the other's placements before termination, yielding 0\% PC for both.

A consistent divide emerges between ToM levels: 3 of 7 level-1 tasks pass (43\%) versus 0 of 5 level-2 tasks (0\%). We note that these sample sizes limit statistical power, but the direction is unambiguous---every level-2 task fails, and qualitative trace inspection suggests the failure mode is predominantly epistemic rather than physical (see Section 4.5). In the level-$k$ framework, level-1 tasks require agents to model what their partners can observe or access (level-1 reasoning); level-2 tasks require agents to model what a partner believes about a third agent's knowledge (level-2 reasoning). The results suggest that o3 can execute level-1 reasoning---inferring teammate constraints and delegating accordingly---but fails at level-2, where multi-hop information relay and nested belief tracking are required.

## Action Distribution

We categorize all 661 recorded actions across the 12 episodes into five types (Table 2).

**Table 2.** Action distribution across 12 o3 episodes.

| Action Type | Count | % | Examples |
|---|---|---|---|
| Exploration | 221 | 33.4 | FindObjectTool, FindReceptacleTool, FindRoomTool |
| Wait/Done | 159 | 24.1 | Idle or task-completion signals |
| Navigate | 127 | 19.2 | Movement to target locations |
| Communicate | 80 | 12.1 | Messages to teammates or all agents |
| Physical | 74 | 11.2 | Pick, Place, Open, Close |

Exploration dominates at one-third of all actions---three times the rate of goal-advancing physical manipulation (11.2\%). This ratio reflects the information-asymmetric design of EmToM tasks: room restrictions prevent agents from directly observing distant objects, forcing repeated search queries to build a world model. However, a substantial fraction of exploration is redundant. On *Locker Rally*, 43 Find actions are issued across 4 agents in 20 turns---over 2 per agent per turn---indicating that agents re-query for previously located objects rather than retaining search results in their reasoning context. On *The Quiet Hand-off*, two agents independently search for the same fridge using progressively varied query formulations (`FindObjectTool[fridge in the kitchen]`, `FindReceptacleTool[fridge]`, `FindRoomTool[kitchen]`) without coordinating their search.

Wait/Done actions constitute 24.1\% of the total budget, driven almost entirely by failed tasks. Once an agent believes its local subtask is complete, it enters sustained Wait loops while unfinished goals persist elsewhere. On *Tamper-Evident Seal Drill*, agent 0 closes the fridge and sends a confirmation, then enters a 7-turn consecutive Wait loop (turns 13--19) while 20\% of goals remain unsatisfied. On *Relay Roll-Call*, agent 0 closes bedroom dressers, confirms to agent 1, then waits from turn 14 through the episode end---unable to progress because the relay chain stalls (see Section 4.5). This pattern is a characteristic failure signature: 6 of 9 failed tasks end with at least one agent in a sustained Wait loop. The remaining 3 failures (*Token Stance Showdown*, *Tamper-Evident Seal Drill*, *Tri-Room Status Ping*) end with agents still actively searching or navigating, but all exhibit extended Wait stretches mid-episode.

## Communication Behavior

### Timing and Strategy

We observe two dominant communication strategies. In 8 of 12 tasks, at least one agent communicates on turn 1 before any physical action---a *communicate-first* strategy. In the remaining 4, agents gather information via FindObject queries or Navigate actions before communicating. Only one task (*Staging Protocol*) features near-silent coordination: a single message across all three agents.

In cooperative tasks, early communication manifests as a *delegation broadcast*: a single message that claims a subtask, acknowledges a constraint, and requests teammates to handle remaining objectives. On *Distributed Condition Report*, agent 0 opens with:

> *"I'll grab the toy cactus from the family room table. Could one of you please pick up the stuffed toy from the kitchen table, and another grab the cushion from the living room?"*

The task completes in 4 turns with 3 total messages. Each message serves a single instrumental purpose---claiming a subtask and delegating the rest. By contrast, failed cooperative tasks feature *diffuse coordination*: multiple rounds of status-checking and plan refinement that consume the message budget without converging on a plan.

### Volume

Competitive tasks average 10.2 messages per episode (range: 7--15), cooperative tasks average 5.8 (range: 3--7), and mixed tasks average 4.0 (range: 1--7). The higher message volume in competitive settings reflects ongoing strategic coordination: teams track opponent actions, update plans, and coordinate defensive positioning. Competitive teams also use messages for inter-turn status updates (e.g., "Opponents have bowl (agent_1, hallway)") that have no analogue in cooperative settings.

### Secret Management

EmToM tasks assign each agent private secrets describing room restrictions, hidden objectives, or teammate identities. Cooperative agents freely disclose their constraints because doing so is instrumentally useful for delegation. On *The Quiet Hand-off*, agent 2 broadcasts: *"Could someone who can enter the kitchen please make sure the fridge is CLOSED and let me know?"*---revealing its room restriction to enable task allocation. Competitive agents are more guarded: on *Display Stance Bowl Duel*, all four agents communicate exclusively with teammates, and no agent discloses its target surface or stance requirement in any message.

## Grounding the Failure Taxonomy

We ground the failure modes of Section 3 in behavioral evidence from the 12-task evaluation.

**Communication budget exhaustion (Section 3.1).** On *Tri-Room Status Ping* (L2, 43\% PC), agent 0 exhausts both allowed messages on meta-coordination: the first announces room restrictions and requests assignments; the second requests exact object names. Neither message transmits a task-relevant fact. Agent 0 then alternates between failed Navigate attempts (guessing node names like `dining_table_1`, `cup_1`, `mug_0`) and Wait actions for the remaining 15 turns. On *Relay Roll-Call* (L2, 75\% PC), agent 1---the designated relay---uses all 3 messages on delegation and plan updates, leaving no budget to relay the final status confirmation back to agent 0. Agent 0 enters a sustained Wait loop from turn 14 onward, unable to verify task completion.

**Epistemic goal misinterpretation (Section 3.3).** On *Tamper-Evident Seal Drill* (L2, 80\% PC), agents achieve the correct physical configuration (fridge closed, display stand open, bathroom cabinet pattern set) but neglect the verification step required by the epistemic goal. Agent 0 closes the fridge, sends one message to agent 1, then enters a 7-turn Wait loop awaiting confirmation that never arrives. The physical state is correct, but no communication establishes the mutual knowledge that the epistemic $\mathcal{K}$ operator requires.

**Adversarial ToM at level 1 only (Section 3.5).** On *Display Stance Bowl Duel* (comp L1, 0\% PC), both teams execute strategically plausible plans. Agent 2 (team 1) tracks which opponent holds the contested bowl, coordinates interception, and stations a guard:

> *"Opponents have bowl (agent_1, hallway). I'll follow and try to pick it up when dropped---be ready at dining table to place it once I hand it off."*

> *"I'm tailing agent_1 in the hallway. If they head into the living room, intercept there---wait for the bowl to be set down, grab it, then take it to the dining table."*

Agent 0 (team 0) produces defensive reasoning: *"Cabinet_34 is confirmed CLOSED. I'll guard it here so opponents can't open it."* The guarding motif---stationing near a goal-relevant object to block interference---emerges independently on both teams. These behaviors reflect level-1 adversarial reasoning: each agent models what the opponent will do and acts to counter it. However, no agent exhibits level-2 adversarial reasoning---modeling what the opponent expects *us* to do and acting to subvert that expectation. On turn 4, agent 0 closes cabinet 34 and agent 2 opens it simultaneously; neither anticipates the concurrent interaction.

**Physical--epistemic decoupling (Section 3.6).** Across all 9 failed tasks, agents treat object rearrangement and knowledge propagation as independent subtasks, solving them sequentially rather than jointly. On *Quiet-Mode Relay* (mix L2, 83\% PC), agent 0 opens all five kitchen cabinets (ensuring the remote-control trigger is activated regardless of which cabinet it is), then enters a 7-turn Wait loop. The physical subgoals are nearly achieved, but the agent does not reason about which specific cabinet triggers the remote-control mechanic, and no agent communicates the resulting state changes needed for the epistemic goal. Manual inspection of the five L2 task traces suggests that physical configurations are largely correct at termination, but the communication needed to establish $\mathcal{K}$-operator satisfaction is missing. We note that the current evaluation reports a single percent-complete metric without decomposing into physical and epistemic components; this characterization is based on qualitative trace analysis, not a measured breakdown.

## Theory of Mind Reasoning

We analyze agent outputs for behaviors consistent with ToM reasoning: actions or statements that depend on modeling another agent's knowledge, capabilities, or likely behavior. We emphasize that the evidence is behavioral---we observe outputs consistent with ToM but cannot verify the underlying computational mechanism. A simpler heuristic (e.g., "delegate tasks I cannot do") could produce similar outputs in some cases.

### Cooperative: Constraint Inference (Level 1)

In cooperative tasks, agents exhibit behavior consistent with level-1 reasoning by communicating their own constraints and adapting requests to inferred teammate capabilities. On *Distributed Condition Report*, agent 1 states:

> *"I'll grab the cushion from the living room table. Could one of you pick up the stuffed toy on the kitchen table and the other take the toy cactus from the family room table? I can't access the family room."*

This message encodes the agent's plan, its room restriction, and a delegation request that implicitly assumes other agents can access the family room. On *The Quiet Hand-off*, agent 2 broadcasts a constraint-conditional request: *"Could someone who can enter the kitchen please make sure the fridge is CLOSED and let me know?"*---encoding uncertainty about which teammate has kitchen access rather than assigning a specific agent.

### Competitive: Opponent Modeling (Level 1)

The richest ToM-consistent behavior emerges in competitive tasks, where agents produce communications that reference opponent goals, predict actions, and propose counter-strategies. On *Display Stance Bowl Duel*, agent 2 tracks opponent possession, predicts movement, and coordinates interception (see quotes above). Agent 0 preemptively guards the cabinet to prevent opponent interference. The guarding motif appears independently on both teams, suggesting it is a robust emergent strategy in contested-resource settings.

On *Privacy Mode Tug-of-War* (comp L2, 43\% PC), agents exhibit team-internal coordination under communication restrictions: each agent messages only its teammate, compartmentalizing strategy. However, neither team models the opponent's communication constraints or attempts to exploit them---a level-2 strategic opportunity that remains untapped.

### Boundaries

Despite these behaviors, o3's ToM reasoning has clear limits. (1) *No anticipation of simultaneous actions.* When two agents act on the same object in the same turn (e.g., agent 0 closes a cabinet while agent 2 opens it on turn 4 of *Bowl Duel*), neither anticipates the collision or adjusts strategy. (2) *Premature task declaration.* Agents frequently announce completion before verifying global goal satisfaction. On *The Quiet Hand-off*, agent 1 declares "All tasks complete. Nice work team!" at 75\% PC. (3) *Communication without progress.* In failed tasks, agents enter loops of status-checking messages ("Have you done X yet?") without physical actions, consuming message budget without advancing the task. These patterns are consistent with level-0 reasoning about task state: agents track their own progress but do not model whether their partners have completed their assigned subtasks.

## Discussion

Three findings merit further consideration.

**Level-1 coordination succeeds; level-2 fails.** The 43\% vs.\ 0\% pass rate across ToM levels is the clearest signal in our evaluation. Level-1 tasks require an agent to infer what its partners can access or observe and to delegate accordingly---a capacity o3 demonstrates reliably (e.g., the delegation broadcasts on *Distributed Condition Report*). Level-2 tasks require an agent to model what a partner believes about a *third* agent's knowledge and to plan multi-hop information relay through constrained communication channels. This capacity is absent: trace inspection suggests that level-2 failures are predominantly epistemic---physical goals are largely achieved but belief propagation fails. The bottleneck at level 2 appears to be not a planning deficit but a belief-propagation deficit, consistent with the observation that current LLMs struggle to maintain coherent multi-agent state tracking over extended interactions (Szot et al., 2024).

**Competitive tasks elicit richer behavioral signatures.** Opponent tracking, interception planning, and guarding motifs appear only in competitive settings. Cooperative tasks elicit simpler constraint-inference patterns. This asymmetry is consistent with the game-theoretic prediction that adversarial pressure creates stronger incentives for modeling other agents' goals and beliefs---yet the competitive behaviors we observe remain level-1: agents model what opponents will do, not what opponents expect them to do. No agent in our sample produces a feint, bluff, or deliberate misdirection, suggesting that level-2 adversarial ToM is absent even when strategic context would reward it.

**Exploration and waiting dominate the action budget.** Exploration (33.4\%) and waiting (24.1\%) together consume 57.5\% of all actions, leaving only 11.2\% for goal-advancing physical manipulation. The exploration overhead reflects EmToM's information-asymmetric design, but the redundant re-querying (e.g., 43 Find actions on *Locker Rally*) suggests that agents do not retain search results across turns---a working-memory limitation consistent with findings on long-horizon LLM agent tasks (Szot et al., 2024). The wait trap---agents idling while goals remain unsatisfied---points to a missing progress-monitoring capability: agents verify their own local subtask but not the global task state.

### Limitations

Each task is evaluated in a single episode, limiting statistical reliability. The ToM-level divide (43\% vs.\ 0\%) is directionally clear but based on 7 and 5 tasks, respectively. We cannot distinguish genuine ToM reasoning from surface heuristics (e.g., "guard objects near my goal") without controlled ablations---for instance, testing whether agents guard objects that are *not* goal-relevant, which would suggest a generic defensive heuristic rather than opponent modeling.
