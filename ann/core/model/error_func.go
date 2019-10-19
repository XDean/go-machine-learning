package model

type ErrorFunc interface {
	CalcError(target, actual Data) (error float64, partial Data) // ∂E/∂a
}
