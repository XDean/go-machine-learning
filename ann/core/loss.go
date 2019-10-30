package core

type LossFunc interface {
	Describable
	CalcLoss(target, actual Data) (error float64, partial Data) // ∂E/∂a
}
