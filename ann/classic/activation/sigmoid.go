package activation

import (
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

func init() {
	persistent.Register(Sigmoid{})
}

type (
	Sigmoid struct{}
)

func (s Sigmoid) Active(input float64) (output, partial float64) {
	output = 1 / (1 + math.Exp(-input))
	return output, output * (1 - output)
}