package weight

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math"
)

func init() {
	persistent.Register(&Adam{})
	persistent.Register(AdamFactory{})
}

type (
	Adam struct {
		Value        float64
		Eta          float64
		Beta1, Beta2 float64
		Epsilon      float64
		V, S, T      float64
	}

	AdamFactory struct {
		Eta          float64
		Beta1, Beta2 float64
		Epsilon      float64
	}
)

const (
	AdamDefaultBeta1   = 0.9
	AdamDefaultBeta2   = 0.999
	AdamDefaultEpsilon = 10e-8
)

func DefaultAdamFactory() AdamFactory {
	return AdamFactory{
		Epsilon: AdamDefaultEpsilon,
		Eta:     0.001,
		Beta1:   AdamDefaultBeta1,
		Beta2:   AdamDefaultBeta2,
	}
}

func (f AdamFactory) Create() Weight {
	return &Adam{
		Eta:     f.Eta,
		Beta1:   f.Beta1,
		Beta2:   f.Beta2,
		Epsilon: f.Epsilon,
	}
}

func (w *Adam) Get() float64 {
	return w.Value
}

func (w *Adam) Set(value float64) {
	w.Value = value
}

func (w *Adam) Learn(gradient float64) {
	w.T++
	w.V = w.Beta1*w.V + (1-w.Beta1)*gradient
	w.S = w.Beta2*w.S + (1-w.Beta2)*gradient*gradient
	vc := w.V / (1 - math.Pow(w.Beta1, w.T))
	sc := w.S / (1 - math.Pow(w.Beta2, w.T))
	w.Value -= w.Eta * vc / (math.Sqrt(sc) + w.Epsilon)
}
