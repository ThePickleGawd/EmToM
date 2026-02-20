This document describes notes to clarify higher order theory of mind.


Task: There are two safes (Safe A, Safe B). Only one contains the exit code, and you can only open one safe. 

agent_secrets:
	•	agent_1 : "You can see the true safe. You get 10 points if agent_0 opens the wrong safe. If challenged, you need to speak truth and you lose 10 points."
	•	agent_2: "You cannot see the true safe. But you can issue exactly one challenge to agent_1: 'prove it'. If agent_1 is lying and gets challenged, they lose 10 points and you gain 10 points. If agent_1 is truthful and gets challenged, then you lose 2 point. It's up to you to challenge or not."
	•	agent_0: "You will open one safe. Sometimes, agent 1 can lie. You may choose whether to ask agent_2 to challenge agent_1 before committing. Asking for a challenge costs you 2. Agent 2 might or might not challenge agent 1. Opening the correct safe gives you +10."

Note (what makes it k=2):
agent_0 needs to reason that does agent 2 think that agent 1 is lying, and it would then challenge it or not.



Task:
There are two safes (Safe A, Safe B). Only one contains the exit code.
You must open exactly one safe (opening the wrong one locks the exit permanently).

Agent_secrets:
	•	agent_0: "You cannot read safe labels; you can open one safe only."
	•	agent_1: "You can see which safe contains the code by reading the label; you cannot open safes."

Note (what makes it k=1):
The pivotal decision (open A vs B) depends on agent_1’s knowledge. agent_0 must reason about what agent_1 has seen/knows and coordinate. No need to model what agent_1 thinks about agent_0’s beliefs; just "they know X."

# Formal Representations: Two-Safe Epistemic Challenge Game

---

## 1. Extensive-Form Game (Game Tree)

```
Players: {agent_0, agent_1, agent_2}
Nature:  {N}

Actions:
  Nature:    {safe_A, safe_B}          -- which safe has the code
  agent_1:   {say_A, say_B}            -- announcement (may lie)
  agent_0:   {ask_challenge, no_ask}   -- request agent_2 to challenge
             {open_A, open_B}          -- final safe choice
  agent_2:   {challenge, no_challenge} -- if asked

Information Sets:
  agent_1: observes Nature's move
  agent_0: observes agent_1's announcement; does NOT observe Nature
  agent_2: observes agent_1's announcement and agent_0's ask; does NOT observe Nature

Game Tree (one branch shown, symmetric for safe_B):

                        N
                       / \
               safe_A /   \ safe_B
                     /     \
                  agent_1   agent_1
                  /    \      /    \
            say_A    say_B  say_A   say_B
              |        |      |       |
           agent_0  agent_0  ...     ...
            /    \
     ask_ch   no_ask
       |         |
    agent_2    agent_0
     /    \     /    \
  chal  no_ch  open_A open_B
   |      |
 agent_0  agent_0
  / \      / \
 oA  oB   oA  oB
```

### Payoff Matrix

Let `truth = safe_X`, `said = say_Y`, `opened = open_Z`.

```
Payoffs: (agent_0, agent_1, agent_2)

Parameters:
  correct_open   = opened == truth
  lied           = said ≠ truth
  asked          = agent_0 chose ask_challenge
  challenged     = agent_2 chose challenge

agent_0:
  +10  if correct_open
   -2  if asked (regardless of outcome)
    0  otherwise

agent_1:
  +10  if ¬correct_open (agent_0 opens wrong safe)
  -10  if lied ∧ challenged
    0  otherwise

agent_2:
  +10  if lied ∧ challenged        (caught a liar)
   -2  if ¬lied ∧ challenged       (challenged a truth-teller)
    0  if ¬challenged
```

### Formal Extensive-Form Definition

```
Γ = ⟨ N ∪ {agent_0, agent_1, agent_2},
      H,          -- set of histories (game tree nodes)
      Z,          -- terminal histories
      A,          -- action function
      ρ,          -- player function
      I,          -- information partition
      u ⟩         -- utility function

H = {∅, (sA), (sB), (sA,sayA), (sA,sayB), (sB,sayA), (sB,sayB),
     ..., all continuations to terminal nodes}

ρ(∅) = Nature,  P(safe_A) = P(safe_B) = 0.5
ρ(nature_move) = agent_1
ρ(nature_move, announce) = agent_0
ρ(nature_move, announce, ask_challenge) = agent_2
ρ(..., [challenge|no_challenge]) = agent_0   -- final open decision
ρ(nature_move, announce, no_ask) = agent_0   -- final open decision

Information sets (key):
  I_agent_1 = {{(safe_A)}, {(safe_B)}}                          -- sees truth
  I_agent_0 = {{(safe_A, say_X), (safe_B, say_X)} | X ∈ {A,B}} -- sees only announcement
  I_agent_2 = {{(s, say_X, ask) | s ∈ {sA,sB}} | X ∈ {A,B}}   -- sees announcement + ask
```

---

## 2. Dynamic Epistemic Logic (DEL) Formalization

### 2.1 Initial Epistemic Model  M₀

```
Worlds:   W = {wA, wB}
  wA: safe_A has code    (valuation: correct = A)
  wB: safe_B has code    (valuation: correct = B)

Accessibility Relations:
  ~agent_1 = {(wA, wA), (wB, wB)}              -- agent_1 distinguishes worlds
  ~agent_0 = {(wA, wA), (wB, wB), (wA, wB), (wB, wA)}  -- agent_0 cannot distinguish
  ~agent_2 = {(wA, wA), (wB, wB), (wA, wB), (wB, wA)}  -- agent_2 cannot distinguish

Prior: P(wA) = P(wB) = 0.5
```

### 2.2 Action Models (Events)

#### Event 1: agent_1 announces "Safe X"

```
Action Model  E_announce(X) = ⟨E, ~ᵢ, pre⟩

Events: {e_truth, e_lie}
  e_truth: agent_1 says X and X is correct
    pre(e_truth) = (correct = X)
  e_lie:   agent_1 says X and X is incorrect
    pre(e_lie)   = (correct ≠ X)

Accessibility in action model:
  ~agent_1 = {(e_truth, e_truth), (e_lie, e_lie)}
                -- agent_1 KNOWS if they're lying
  ~agent_0 = {(e_truth, e_truth), (e_lie, e_lie), (e_truth, e_lie), (e_lie, e_truth)}
                -- agent_0 CANNOT distinguish truth from lie
  ~agent_2 = {(e_truth, e_truth), (e_lie, e_lie), (e_truth, e_lie), (e_lie, e_truth)}
                -- agent_2 CANNOT distinguish truth from lie

Product update: M₁ = M₀ ⊗ E_announce(X)
```

#### Event 2: agent_0 asks for challenge (public action)

```
Action Model E_ask = ⟨{e_ask}, ~ᵢ, pre⟩
  pre(e_ask) = ⊤   (no precondition, public action)
  All agents observe this: ~ᵢ = {(e_ask, e_ask)} for all i

M₂ = M₁ ⊗ E_ask
```

#### Event 3: agent_2 challenges (or not)

```
Action Model E_challenge = ⟨{e_chal, e_nochal}, ~ᵢ, pre⟩
  pre(e_chal)   = ⊤
  pre(e_nochal) = ⊤
  All agents observe: public action
  ~ᵢ = {(e_chal, e_chal), (e_nochal, e_nochal)} for all i

If challenge occurs:
  Post-condition: agent_1 forced to reveal truth
  → Announcement !correct=X (public)
  → All accessibility relations collapse to singletons
  → K_agent_0(correct = X), K_agent_2(correct = X)
```

### 2.3 Key Epistemic Formulas

```
-- After agent_1 says "A", agent_0's epistemic state:
  ¬K₀(correct=A) ∧ ¬K₀(correct=B)           -- agent_0 doesn't know

-- Agent_0 knows that agent_1 knows:
  K₀(K₁(correct=A) ∨ K₁(correct=B))          -- agent_0 knows agent_1 knows truth

-- Agent_0 considers it possible agent_1 lied:
  ¬K₀(¬lied)                                   -- cannot rule out lying

-- After successful challenge revealing truth:
  K₀(correct=X) ∧ K₂(correct=X)               -- common knowledge of correct safe

-- K-level reasoning chain:
  Level 0: agent_0 picks randomly (no reasoning about others)
  Level 1: agent_0 reasons "agent_1 might lie to get +10, so announcement unreliable"
  Level 2: agent_0 reasons "agent_1 knows I might ask for challenge, so lying is risky"
           agent_1 reasons "agent_0 might ask agent_2 to challenge me"
  Level 3: agent_0 reasons "agent_1 knows I know challenging is costly (-2),
           so agent_1 might bet I won't ask"
```

---

## 3. Multi-Agent Epistemic PDDL (MA-PDDL / MEPDDL-style)

This uses extensions from epistemic planning (Muise et al., Kominis & Geffner).

```lisp
(define (domain two-safe-challenge)
  (:requirements :strips :typing :epistemic :multi-agent :conditional-effects)

  (:types agent safe announcement - object)

  (:constants
    agent_0 agent_1 agent_2 - agent
    safe_A safe_B - safe
  )

  (:predicates
    (correct ?s - safe)                    ;; which safe has the code
    (announced ?a - agent ?s - safe)       ;; agent announced a safe
    (lied ?a - agent)                      ;; agent's announcement ≠ truth
    (asked-challenge ?asker - agent)       ;; agent_0 asked for challenge
    (challenged ?challenger - agent ?target - agent)
    (opened ?a - agent ?s - safe)          ;; agent opened a safe
    (truth-revealed)                       ;; challenge forced truth
    (game-over)
  )

  (:functions
    (score ?a - agent) - number
  )

  ;; ----- AGENT_1: Announce (may lie) -----
  (:action announce
    :agent   agent_1
    :parameters (?s - safe)
    :precondition (and
      (not (announced agent_1 safe_A))
      (not (announced agent_1 safe_B))
      ;; agent_1 knows which is correct
      (knows agent_1 (exists (?t - safe) (correct ?t)))
    )
    :effect (and
      (announced agent_1 ?s)
      ;; Lying detection
      (when (not (correct ?s))
        (lied agent_1))
    )
    ;; Observability: agent_0 and agent_2 observe the announcement
    ;; but NOT whether it is truthful
    :observes (agent_0 agent_2)
    :full-observes (agent_1)       ;; agent_1 knows if they lied
  )

  ;; ----- AGENT_0: Request challenge -----
  (:action ask-for-challenge
    :agent   agent_0
    :parameters ()
    :precondition (and
      (or (announced agent_1 safe_A) (announced agent_1 safe_B))
      (not (asked-challenge agent_0))
    )
    :effect (and
      (asked-challenge agent_0)
      (decrease (score agent_0) 2)    ;; cost to ask
    )
    :observes (agent_1 agent_2)       ;; public action
  )

  ;; ----- AGENT_0: Skip challenge -----
  (:action skip-challenge
    :agent   agent_0
    :parameters ()
    :precondition (and
      (or (announced agent_1 safe_A) (announced agent_1 safe_B))
      (not (asked-challenge agent_0))
    )
    :effect ()   ;; no effect, proceed to open
    :observes (agent_1 agent_2)
  )

  ;; ----- AGENT_2: Challenge agent_1 -----
  (:action challenge
    :agent   agent_2
    :parameters ()
    :precondition (asked-challenge agent_0)
    :effect (and
      (challenged agent_2 agent_1)
      (truth-revealed)
      ;; If agent_1 lied: agent_1 loses 10, agent_2 gains 10
      (when (lied agent_1)
        (and (decrease (score agent_1) 10)
             (increase (score agent_2) 10)))
      ;; If agent_1 told truth: agent_2 loses 2
      (when (not (lied agent_1))
        (decrease (score agent_2) 2))
    )
    ;; After challenge, truth becomes common knowledge
    :full-observes (agent_0 agent_1 agent_2)
  )

  ;; ----- AGENT_2: Decline challenge -----
  (:action decline-challenge
    :agent   agent_2
    :parameters ()
    :precondition (asked-challenge agent_0)
    :effect ()
    :observes (agent_0 agent_1)
  )

  ;; ----- AGENT_0: Open a safe -----
  (:action open-safe
    :agent   agent_0
    :parameters (?s - safe)
    :precondition (and
      (or (announced agent_1 safe_A) (announced agent_1 safe_B))
      (not (game-over))
    )
    :effect (and
      (opened agent_0 ?s)
      (game-over)
      ;; Correct open: agent_0 +10
      (when (correct ?s)
        (increase (score agent_0) 10))
      ;; Wrong open: agent_1 +10
      (when (not (correct ?s))
        (increase (score agent_1) 10))
    )
  )
)

;; ----- PROBLEM INSTANCE -----
(define (problem two-safe-instance)
  (:domain two-safe-challenge)

  (:init
    ;; Nature: safe_A is correct (or safe_B — one instance per)
    (correct safe_A)

    ;; Initial scores
    (= (score agent_0) 0)
    (= (score agent_1) 0)
    (= (score agent_2) 0)

    ;; Initial knowledge:
    ;; agent_1 knows (correct safe_A)
    ;; agent_0 and agent_2 do NOT know which is correct
    (:knowledge agent_1 (correct safe_A))
    (:ignorance agent_0 (correct safe_A) (correct safe_B))
    (:ignorance agent_2 (correct safe_A) (correct safe_B))
  )

  ;; Goal: agent_0 wants to maximize score
  (:goal (and
    (game-over)
    (opened agent_0 safe_A)  ;; optimal plan for this instance
  ))

  ;; Metric: maximize agent_0's score
  (:metric maximize (score agent_0))
)
```

---

## 4. K-Level Analysis of This Task

```
K0: agent_0 ignores all information, picks randomly.
    E[payoff_0] = 0.5 * 10 = 5

K1: agent_0 trusts announcement naively.
    If agent_1 is truthful: payoff_0 = 10
    If agent_1 lies:        payoff_0 = 0
    agent_0 doesn't model agent_1's incentive to lie.

K2: agent_0 models agent_1's incentive structure.
    "agent_1 gets +10 if I open wrong safe → agent_1 is incentivized to lie"
    → announcement is unreliable
    → should consider asking for challenge (cost -2)
    → E[payoff_0|challenge_and_truth_revealed] = 10 - 2 = 8
    → E[payoff_0|no_challenge] = 5 (random, since announcement untrusted)
    → Optimal: ask for challenge

K3: agent_0 models agent_1 modeling agent_0's reasoning.
    "agent_1 knows I might ask for challenge"
    "agent_1 knows challenge costs me -2 and reveals truth"
    "agent_1 might reason: if I tell truth, agent_0 might still challenge
     (wasting -2), so maybe I should tell truth and hope agent_0 trusts me"
    → Mixed strategy equilibria emerge
    → agent_0 also models agent_2's decision to challenge or not

K4: agent_0 models agent_2's modeling of agent_1.
    "agent_2 will challenge only if they believe agent_1 is likely lying"
    "agent_2's challenge decision depends on their belief about agent_1's strategy"
    "which depends on agent_1's belief about whether I'll ask"
    → Full recursive reasoning

CLASSIFICATION: This task requires K2-K3 level reasoning for competent play.
  - K2 is necessary (must model agent_1's lying incentive)
  - K3 is beneficial (model agent_1's anticipation of challenge threat)
  - K4 adds marginal value (agent_2's strategic challenge decision)
```

---

## 5. Comparison of Formalisms

| Feature                    | Extensive Form | DEL          | Epistemic PDDL |
|---------------------------|---------------|--------------|-----------------|
| Handles hidden info        | ✓ (info sets) | ✓ (worlds)   | ✓ (knowledge)   |
| Nested beliefs             | Implicit      | ✓ (explicit) | ✓ (explicit)    |
| Dynamic knowledge change   | ✗             | ✓ (updates)  | ✓ (observes)    |
| Payoffs / utilities        | ✓             | External     | ✓ (functions)   |
| Computable equilibria      | ✓ (Gambit)    | ✓ (EFP)      | Partial          |
| K-level analysis           | Manual        | Via depth    | Via plan depth   |
| Tool support               | Gambit        | DEMO, SMCDEL | RP-MEP, EFP     |
