package classic

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math"
)

type (
	Activation interface {
		persistent.Persistent
		Active(input float64) (output, partial float64)
	}
)

func init() {
	persistent.Register(func() persistent.Persistent { return Sigmoid{} })
	persistent.Register(func() persistent.Persistent { return ReLU{} })
	persistent.Register(func() persistent.Persistent { return Linear{} })
}

// built-in
type (
	Sigmoid struct {
		persistent.TypePersistent
	}

	ReLU struct {
		persistent.TypePersistent
	}

	Linear struct {
		persistent.TypePersistent
	}
)

func (s Sigmoid) Name() string {
	return "Activation-Sigmoid"
}

func (s Sigmoid) Active(input float64) (output, partial float64) {
	output = 1 / (1 + math.Exp(-input))
	return output, output * (1 - output)
}

func (r ReLU) Name() string {
	return "Activation-ReLU"
}

func (r ReLU) Active(input float64) (output, partial float64) {
	if input > 0 {
		return input, 1
	} else {
		return 0, 0
	}
}

func (l Linear) Name() string {
	return "Activation-Linear"
}

func (l Linear) Active(input float64) (output, partial float64) {
	return input, 1
}
