package classic

import (
	. "github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/model/persistent"
)

type SquareError struct {
	persistent.TypePersistent
}

func (s SquareError) Name() string {
	return "SquareError"
}

func (s SquareError) Error(target, actual Data) (error float64, partial Data) {
	partial = NewData(actual.GetSize()...)
	target.ForEach(func(index []uint, t float64) {
		a := actual.GetValue(index...)
		error += (t - a) * (t - a) / 2
		partial = partial.SetValue(a-t, index...)
	})
	return
}
