package classic

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math/rand"
)

type WeightInit interface {
	Init(data data.Data) data.Data
}

func init() {
	persistent.Register(&RandomInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) Init(data data.Data) data.Data {
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
