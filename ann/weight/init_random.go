package weight

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math/rand"
)

func init() {
	persistent.Register(RandomInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r RandomInit) Generate(count int) func() float64 {
	return func() float64 {
		v := rand.Float64()
		if !r.PositiveOnly {
			v = (v - 0.5) * 2
		}
		v *= r.Range
		return v
	}
}
