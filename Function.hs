module Function where

import Numeric.LinearAlgebra

sigmoid :: R -> R
sigmoid x = 1.0 / (1.0 + exp (-x))

sigmoid' :: R -> R
sigmoid' x = sigmoid x * (1.0 - sigmoid x)

relu :: R -> R
relu x
  | x > 0     = x
  | otherwise = 0

relu' :: R -> R
relu' x
  | x > 0     = 1
  | otherwise = 0

softmax :: Vector R -> Vector R
softmax v = expv / scalar w
  where v'   = maxElement v
        expv = cmap exp (v - scalar v')
        w    = sumElements expv

