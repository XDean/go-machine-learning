package weight

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math"
)

func init() {
	persistent.Register(&AdaGrad{})
	persistent.Register(AdaGradFactory{})
}

type (
	AdaGrad struct {
		Value   float64
		Eta     float64
		Epsilon float64
		Sum     float64
	}
	AdaGradFactory struct {
		Eta     float64
		Epsilon float64
	}
)

const (
	AdaGradDefaultEpsilon = 10e-6
)

func DefaultAdaGradFactory() AdaGradFactory {
	return AdaGradFactory{
		Epsilon: AdaGradDefaultEpsilon,
		Eta:     0.1,
	}
}

func (f AdaGradFactory) Create() Weight {
	return &AdaGrad{
		Eta:     f.Eta,
		Epsilon: f.Epsilon,
	}
}

func (w *AdaGrad) Get() float64 {
	return w.Value
}

func (w *AdaGrad) Set(value float64) {
	w.Value = value
}

func (w *AdaGrad) Learn(gradient float64) {
	w.Sum += gradient * gradient
	w.Value -= w.Eta * gradient / (math.Sqrt(w.Sum) + w.Epsilon)
}
