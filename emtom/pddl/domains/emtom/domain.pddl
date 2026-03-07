; Generated from domain.py -- do not edit manually
(define (domain emtom)
  (:requirements :strips :typing :conditional-effects)
  (:types agent room item furniture - object)
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
    (can_communicate ?from - agent ?to - agent)
  )

(:action open
  :parameters (?a - agent ?f - furniture ?r - room)
  :precondition (and (agent_in_room ?a ?r) (is_in_room ?f ?r) (is_closed ?f) (not (is_locked_permanent ?f)))
  :effect (and (is_open ?f) (not (is_closed ?f)) (when (is_inverse ?f) (is_closed ?f)) (when (is_inverse ?f) (not (is_open ?f))) (forall (?g - furniture) (when (mirrors ?f ?g) (is_open ?g))) (forall (?g - furniture) (when (controls ?f ?g) (is_unlocked ?g))))
)

(:action close
  :parameters (?a - agent ?f - furniture ?r - room)
  :precondition (and (agent_in_room ?a ?r) (is_in_room ?f ?r) (is_open ?f))
  :effect (and (is_closed ?f) (not (is_open ?f)) (when (is_inverse ?f) (is_open ?f)) (when (is_inverse ?f) (not (is_closed ?f))) (forall (?g - furniture) (when (mirrors ?f ?g) (is_closed ?g))))
)

(:action navigate
  :parameters (?a - agent ?r - room)
  :precondition (not (is_restricted ?a ?r))
  :effect (and (agent_in_room ?a ?r) (forall (?old - room) (when (agent_in_room ?a ?old) (and (agent_in_room ?a ?r) (not (agent_in_room ?a ?old))))))
)

(:action pick
  :parameters (?a - agent ?x - object ?r - room)
  :precondition (and (agent_in_room ?a ?r) (is_in_room ?x ?r))
  :effect (is_held_by ?x ?a)
)

(:action place
  :parameters (?a - agent ?x - object ?f - furniture ?r - room)
  :precondition (and (is_held_by ?x ?a) (agent_in_room ?a ?r) (is_in_room ?f ?r))
  :effect (and (not (is_held_by ?x ?a)) (is_on_top ?x ?f) (is_inside ?x ?f) (is_in_room ?x ?r))
)

(:action use_item
  :parameters (?a - agent ?i - item ?f - furniture ?r - room)
  :precondition (and (has_item ?a ?i) (requires_item ?f ?i) (agent_in_room ?a ?r) (is_in_room ?f ?r))
  :effect (is_unlocked ?f)
)
)
