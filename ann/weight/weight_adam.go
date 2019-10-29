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
)

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
