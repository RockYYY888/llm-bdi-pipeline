(define (domain blocksworld)
  (:requirements :equality :typing)
  (:types block)
  (:predicates (holding ?b - block) (handempty) (ontable ?b - block) (on ?b1 ?b2 - block) (clear ?b - block))
  (:action pick-up
    :parameters (?b1 ?b2 - block)
    :precondition (and (not (= ?b1 ?b2)) (handempty) (clear ?b1) (on ?b1 ?b2))
    :effect (and (holding ?b1) (clear ?b2) (not (handempty)) (not (clear ?b1)) (not (on ?b1 ?b2)))
  )
  (:action pick-up-from-table
    :parameters (?b - block)
    :precondition (and (handempty) (clear ?b) (ontable ?b))
    :effect (and (holding ?b) (not (handempty)) (not (ontable ?b)))
  )
  (:action put-on-block
    :parameters (?b1 ?b2 - block)
    :precondition (and (not (= ?b1 ?b2)) (holding ?b1) (clear ?b2))
    :effect (and (on ?b1 ?b2) (handempty) (clear ?b1) (not (holding ?b1)) (not (clear ?b2)))
  )
  (:action put-down
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (ontable ?b) (handempty) (clear ?b) (not (holding ?b)))
  )

)
