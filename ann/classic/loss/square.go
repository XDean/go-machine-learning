package loss

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(SquareError{})
}

type SquareError struct{}

func (s SquareError) CalcLoss(target, actual Data) (error float64, partial Data) {
	partial = NewData(actual.Size)

	target.ForEachIndex(func(i, j, k int, t float64) {
		a := actual.Value[i][j][k]
		error += (t - a) * (t - a) / 2
		partial.Value[i][j][k] = a - t
	})
	return
}
