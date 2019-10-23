package activation

import "github.com/XDean/go-machine-learning/ann/core/persistent"

func init() {
	persistent.Register(ReLU{})
	persistent.Register(LeakReLU{})
}

type (
	ReLU     struct{}
	LeakReLU struct {
		Rate float64
	}
)

func (r ReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return 0, 0
	}
}

func (r LeakReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return input * r.Rate, r.Rate
	}
}
