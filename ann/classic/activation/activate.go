package activation

import (
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

type (
	Activation interface {
		Active(input float64) (output, partial float64)
	}
)

func init() {
	persistent.Register(Sigmoid{})
	persistent.Register(ReLU{})
	persistent.Register(Linear{})
}

// built-in
type (
	Sigmoid struct{}
	Linear  struct{}
)

func (s Sigmoid) Active(input float64) (output, partial float64) {
	output = 1 / (1 + math.Exp(-input))
	return output, output * (1 - output)
}

func (l Linear) Active(input float64) (output, partial float64) {
	return input, 1
}
