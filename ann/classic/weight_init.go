package classic

import (
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math/rand"
)

type WeightInit interface {
	Init(data base.Data) base.Data
}

func init() {
	persistent.Register(RandomInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) Init(data base.Data) base.Data {
	data.ForEach(func(index []int, value float64) {
		v := rand.Float64()
		if !r.PositiveOnly {
			v = (v - 0.5) * 2
		}
		v *= r.Range
		data.SetValue(v, index...)
	})
	return data
}
