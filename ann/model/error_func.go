package model

type ErrorFunc func(target, actual Data) (error float64, partial Data) // ∂E/∂a

func SquareError() ErrorFunc {
	return func(target, actual Data) (error float64, partial Data) {
		partial = NewData(actual.GetSize()...)
		target.ForEach(func(index []uint, t float64) {
			a := actual.GetValue(index...)
			error += (t - a) * (t - a) / 2
			partial = partial.SetValue(a-t, index...)
		})
		return
	}
}
