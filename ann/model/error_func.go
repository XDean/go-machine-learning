package model

type ErrorFunc func(target, actual Data) (error float64, partial Data) // ∂E/∂a
