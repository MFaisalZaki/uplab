(define (domain plain)
  (:requirements :strips :typing)
  (:types loc)
  (:predicates (at ?l - loc) (road ?from ?to - loc))
  (:action move
    :parameters (?from ?to - loc)
    :precondition (and (at ?from) (road ?from ?to))
    :effect (and (not (at ?from)) (at ?to))))
