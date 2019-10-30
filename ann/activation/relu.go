package activation

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(ReLU{})
	persistent.Register(LeakReLU{})
}

type (
	ReLU     struct{}
	LeakReLU struct {
		Ratio float64
	}
)

func (r ReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return 0, 0
	}
}

func (r ReLU) Desc() core.Desc {
	return core.SimpleDesc{Name: "ReLU"}
}

func (r LeakReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return input * r.Ratio, r.Ratio
	}
}

func (r LeakReLU) Desc() core.Desc {
	return core.SimpleDesc{
		Name: "LeakReLU",
		Params: map[string]interface{}{
			"Leak Ratio": r.Ratio,
		},
	}
}
