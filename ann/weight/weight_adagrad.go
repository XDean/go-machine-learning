package weight

import "math"

type (
	AdaGrad struct {
		Value   float64
		Eta     float64
		Epsilon float64
		Sum     float64
	}
)

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
