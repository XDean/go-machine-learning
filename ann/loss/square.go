package loss

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(Square{})
}

type Square struct{}

func (s Square) CalcLoss(target, actual core.Data) (error float64, partial core.Data) {
	partial = core.NewData(actual.Size)

	target.ForEachIndex(func(i, j, k int, t float64) {
		a := actual.Value[i][j][k]
		error += (t - a) * (t - a) / 2
		partial.Value[i][j][k] = a - t
	})
	return
}
