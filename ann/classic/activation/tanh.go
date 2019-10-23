package activation

import (
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

func init() {
	persistent.Register(Tanh{})
}

type Tanh struct{}

func (t Tanh) Active(input float64) (output, partial float64) {
	output = math.Tanh(input)
	return output, 1 - output*output
}
