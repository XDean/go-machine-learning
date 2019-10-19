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
	persistent.Register(&NormalInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) Init(data data.Data) data.Data {
	data.ForEachIndex(func(index []int, value float64) {
		v := rand.Float64()
		if !r.PositiveOnly {
			v = (v - 0.5) * 2
		}
		v *= r.Range
		data.SetValue(v, index)
	})
	return data
}

type NormalInit struct {
	Mean float64
	Std  float64
}

func (r *NormalInit) Init(data data.Data) data.Data {
	data.MapIndex(func(index []int, _ float64) float64 {
		v := rand.NormFloat64()
		return (v * r.Std) + r.Mean
	})
	return data
}
