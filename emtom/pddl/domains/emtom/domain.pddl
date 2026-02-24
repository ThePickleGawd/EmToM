(define (domain emtom)
  (:requirements :strips :typing :epistemic)
  (:types agent object furniture - object room item)
  (:predicates
    (is_on_top ?x - object ?y - furniture)
    (is_inside ?x - object ?y - furniture)
    (is_in_room ?x - object ?r - room)
    (is_on_floor ?x - object)
    (is_next_to ?x - object ?y - object)
    (is_open ?f - furniture)
    (is_closed ?f - furniture)
    (is_clean ?x - object)
    (is_dirty ?x - object)
    (is_filled ?x - object)
    (is_empty ?x - object)
    (is_powered_on ?x - object)
    (is_powered_off ?x - object)
    (is_unlocked ?f - furniture)
    (is_held_by ?x - object ?a - agent)
    (agent_in_room ?a - agent ?r - room)
    (has_item ?a - agent ?i - item)
    (has_at_least ?a - agent ?i - item)
    (has_most ?a - agent ?i - item)
    (is_inverse ?f - furniture)
    (mirrors ?f1 - furniture ?f2 - furniture)
    (controls ?f1 - furniture ?f2 - furniture)
    (is_restricted ?a - agent ?r - room)
    (is_locked_permanent ?f - furniture)
    (requires_item ?f - furniture ?i - item)
  )

(:action open
  :parameters (?a - agent ?f - furniture)
  :precondition (and (is_closed ?f) (not (is_locked_permanent ?f)))
  :effect (and (is_open ?f) (not (is_closed ?f)) (when (is_inverse ?f) (is_closed ?f)) (when (mirrors ?f ?g) (is_open ?g)) (when (controls ?f ?g) (is_unlocked ?g)))
)

(:action close
  :parameters (?a - agent ?f - furniture)
  :precondition (is_open ?f)
  :effect (and (is_closed ?f) (not (is_open ?f)) (when (is_inverse ?f) (is_open ?f)) (when (mirrors ?f ?g) (is_closed ?g)))
)

(:action navigate
  :parameters (?a - agent ?r - room)
  :precondition (not (is_restricted ?a ?r))
  :effect (agent_in_room ?a ?r)
)

(:action pick
  :parameters (?a - agent ?x - object)
  :precondition ()
  :effect (is_held_by ?x ?a)
)

(:action place
  :parameters (?a - agent ?x - object ?f - furniture)
  :precondition (is_held_by ?x ?a)
  :effect (and (not (is_held_by ?x ?a)) (is_on_top ?x ?f))
)

(:action communicate
  :parameters (?from - agent ?to - agent)
  :precondition ()
  :effect (and )
)

(:action wait
  :parameters (?a - agent)
  :precondition ()
  :effect (and )
)

(:action use_item
  :parameters (?a - agent ?i - item ?f - furniture)
  :precondition (and (has_item ?a ?i) (requires_item ?f ?i))
  :effect (is_unlocked ?f)
)
)
