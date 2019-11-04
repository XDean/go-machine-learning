package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"math"
)

type Loss struct {
	J float64
}

func (l *Loss) Desc() core.Desc {
	return core.SimpleDesc{Name: "LeNet-5"}
}

func (l *Loss) CalcLoss(target, actual core.Data) (error float64, partial core.Data) {
	expect := 0
	sum := math.Exp(-l.J)
	for i, v := range target.Value[0][0] {
		if v == 1 {
			expect = i
		} else {
			sum += math.Exp(-actual.Value[0][0][i])
		}
	}
	error = actual.Value[0][0][expect] + math.Log(sum)
	partial = core.NewData(core.Size{1, 1, 10})
	partial.Value[0][0][expect] = 1
	partial.MapIndex(func(_, _, i int, _ float64) float64 {
		if i == expect {
			return 1
		} else {
			return -math.Exp(-actual.Value[0][0][i]) / sum
		}
	})
	return
}
