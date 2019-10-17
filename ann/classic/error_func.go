package classic

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(SquareError{})
}

type SquareError struct{}

func (s SquareError) CalcError(target, actual data.Data) (error float64, partial data.Data) {
	partial = data.NewData(actual.GetSize()...)
	target.ForEachIndex(func(index []int, t float64) {
		a := actual.GetValue(index...)
		error += (t - a) * (t - a) / 2
		partial = partial.SetValue(a-t, index...)
	})
	return
}
