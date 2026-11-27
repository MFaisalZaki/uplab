(define (domain roads)
  (:requirements :strips :typing)
  (:types location)
  (:predicates (at ?l - location) (road ?a - location ?b - location))
  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?from) (road ?from ?to))
    :effect (and (not (at ?from)) (at ?to))))
