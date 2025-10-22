(define (problem simple-stack)
  (:domain blocksworld)
  (:objects a b)
  (:init
    (ontable a)
    (ontable b)
    (clear a)
    (clear b)
    (handempty)
  )
  (:goal (and
    (on a b)
  ))
)
