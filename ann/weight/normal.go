package weight

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math/rand"
)

func init() {
	persistent.Register(&NormalInit{})
}

type NormalInit struct {
	Mean float64
	Std  float64
}

func (r *NormalInit) InitOne() float64 {
	v := rand.NormFloat64()
	return (v * r.Std) + r.Mean
}

func (r *NormalInit) InitData(data core.Data) {
	data.Map(func(_ float64) float64 {
		return r.InitOne()
	})
}
