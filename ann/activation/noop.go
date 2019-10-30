package activation

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(NoOp{})
}

type NoOp struct{}

func (l NoOp) Active(input float64) (output, partial float64) {
	return input, 1
}

func (l NoOp) Desc() core.Desc {
	return core.SimpleDesc{
		Name: "no-op",
	}
}
