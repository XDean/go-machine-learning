package weight

import "math"

type (
	Adam struct {
		Value        float64
		Eta          float64
		Beta1, Beta2 float64
		V, S, T      float64
		Epsilon      float64
	}

	AdamFactory struct {
		Beta1, Beta2 float64
		Epsilon      float64
	}
)

const (
	AdamDefaultBeta1   = 0.9
	AdamDefaultBeta2   = 0.999
	AdamDefaultEpsilon = 10e-8
)

func (f AdamFactory) Create() Weight {
	return &Adam{
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
