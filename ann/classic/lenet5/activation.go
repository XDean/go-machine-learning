package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/core"
)

const MagicA = 1.7159

type Activation struct {
	activation.Tanh
}

func (a Activation) Active(input float64) (output, partial float64) {
	output, partial = a.Tanh.Active(input)
	output *= MagicA
	partial *= MagicA
	return
}

func (a Activation) Desc() core.Desc {
	return core.SimpleDesc{Name: "Tanh * 1.7159"}
}
