package weight

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math/rand"
)

func init() {
	persistent.Register(&RandomInit{})
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) InitOne() float64 {
	v := rand.Float64()
	if !r.PositiveOnly {
		v = (v - 0.5) * 2
	}
	v *= r.Range
	return v
}

func (r *RandomInit) InitData(data Data) {
	data.Map(func(_ float64) float64 {
		return r.InitOne()
	})
}