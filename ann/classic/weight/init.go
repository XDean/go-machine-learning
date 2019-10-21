package weight

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math/rand"
)

type Init interface {
	Init(data Data) Data
}

func init() {
	persistent.Register(&RandomInit{})
	persistent.Register(&NormalInit{})
	persistent.Register(&ZeroInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) Init(data Data) Data {
	data.Map(func(value float64) float64 {
		v := rand.Float64()
		if !r.PositiveOnly {
			v = (v - 0.5) * 2
		}
		v *= r.Range
		return v
	})
	return data
}

type NormalInit struct {
	Mean float64
	Std  float64
}

func (r *NormalInit) Init(data Data) Data {
	data.Map(func(_ float64) float64 {
		v := rand.NormFloat64()
		return (v * r.Std) + r.Mean
	})
	return data
}

type ZeroInit struct {
}

func (r *ZeroInit) Init(data Data) Data {
	data.Map(func(_ float64) float64 {
		return 0
	})
	return data
}
