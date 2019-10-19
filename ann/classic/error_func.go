package classic

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(SquareError{})
}

type SquareError struct{}

func (s SquareError) CalcError(target, actual Data) (error float64, partial Data) {
	partial = NewData(actual.Size)

	target.ForEachIndex(func(index []int, t float64) {
		a := actual.GetValue(index)
		error += (t - a) * (t - a) / 2
		partial.SetValue(a-t, index)
	})
	return
}
