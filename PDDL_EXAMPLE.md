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