package classic

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
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
	ReLU    struct{}
	Linear  struct{}
)

func (s Sigmoid) Active(input float64) (output, partial float64) {
	output = 1 / (1 + math.Exp(-input))
	return output, output * (1 - output)
}

func (r ReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return 0, 0
	}
}

func (l Linear) Active(input float64) (output, partial float64) {
	return input, 1
}
