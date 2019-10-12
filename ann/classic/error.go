package classic

import (
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(func() persistent.Persistent {
		return SquareError{}
	})
}

type SquareError struct {
	persistent.TypePersistent
}

func (s SquareError) Name() string {
	return "SquareError"
}

func (s SquareError) CalcError(target, actual base.Data) (error float64, partial base.Data) {
	partial = base.NewData(actual.GetSize()...)
	target.ForEach(func(index []int, t float64) {
		a := actual.GetValue(index...)
		error += (t - a) * (t - a) / 2
		partial = partial.SetValue(a-t, index...)
	})
	return
}
