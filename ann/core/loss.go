package core

type LossFunc interface {
	CalcLoss(target, actual Data) (error float64, partial Data) // ∂E/∂a
}
