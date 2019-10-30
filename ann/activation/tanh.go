package activation

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
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

func (t Tanh) Desc() core.Desc {
	return core.SimpleDesc{Name: "Tanh"}
}
