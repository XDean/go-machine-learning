package loss

import (
	"github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

func init() {
	persistent.Register(CrossEntropy{})
}

type CrossEntropy struct {
}

func (c CrossEntropy) CalcLoss(target, actual model.Data) (error float64, partial model.Data) {
	partial = model.NewData(actual.Size)
	count := float64(target.Size.GetCount())
	actual.ForEachIndex(func(i, j, k int, a float64) {
		y := target.Value[i][j][k]
		error -= (y*math.Log(a) + (1-y)*math.Log(1-a)) / count
		partial.Value[i][j][k] = ((1-y)/(1-a) - y/a) / count
	})
	return
}
