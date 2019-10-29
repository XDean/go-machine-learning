package weight

import (
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

func (r *NormalInit) Generate(count int) func() float64 {
	return func() float64 {
		v := rand.NormFloat64()
		return (v * r.Std) + r.Mean
	}
}
