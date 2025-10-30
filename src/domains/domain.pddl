(define (domain blocksworld)
  (:requirements :non-deterministic :equality :typing)
  (:types block)
  (:predicates (holding ?b - block) (handempty) (ontable ?b - block) (on ?b1 ?b2 - block) (clear ?b - block))
  (:action pick-up
    :parameters (?b1 ?b2 - block)
    :precondition (and (not (= ?b1 ?b2)) (handempty) (clear ?b1) (on ?b1 ?b2))
    :effect
      (oneof 
        (and (holding ?b1) (clear ?b2) (not (handempty)) (not (clear ?b1)) (not (on ?b1 ?b2)))
        (and (clear ?b2) (ontable ?b1) (not (on ?b1 ?b2))))
  )
  (:action pick-up-from-table
    :parameters (?b - block)
    :precondition (and (handempty) (clear ?b) (ontable ?b))
    :effect (oneof (and) (and (holding ?b) (not (handempty)) (not (ontable ?b))))
  )
  (:action put-on-block
    :parameters (?b1 ?b2 - block)
    :precondition (and (holding ?b1) (clear ?b2))
    :effect (oneof (and (on ?b1 ?b2) (handempty) (clear ?b1) (not (holding ?b1)) (not (clear ?b2)))
                   (and (ontable ?b1) (handempty) (clear ?b1) (not (holding ?b1))))
  )
  (:action put-down
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (ontable ?b) (handempty) (clear ?b) (not (holding ?b)))
  )
  (:action pick-tower
    :parameters (?b1 ?b2 ?b3 - block)
    :precondition (and (handempty) (on ?b1 ?b2) (on ?b2 ?b3))
    :effect
      (oneof (and) (and (holding ?b2) (clear ?b3) (not (handempty)) (not (on ?b2 ?b3))))
  )
  (:action put-tower-on-block
    :parameters (?b1 ?b2 ?b3 - block)
    :precondition (and (holding ?b2) (on ?b1 ?b2) (clear ?b3))
    :effect (oneof (and (on ?b2 ?b3) (handempty) (not (holding ?b2)) (not (clear ?b3)))
                   (and (ontable ?b2) (handempty) (not (holding ?b2))))
  )
  (:action put-tower-down
    :parameters (?b1 ?b2 - block)
    :precondition (and (holding ?b2) (on ?b1 ?b2))
    :effect (and (ontable ?b2) (handempty) (not (holding ?b2)))
  )
)
