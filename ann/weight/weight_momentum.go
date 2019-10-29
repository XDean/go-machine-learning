package weight

type Momentum struct {
	Value    float64
	Eta      float64
	Gamma    float64
	Velocity float64
}

func (w *Momentum) Get() float64 {
	return w.Value
}

func (w *Momentum) Set(value float64) {
	w.Value = value
}

func (w *Momentum) Learn(gradient float64) {
	w.Velocity = w.Gamma*w.Velocity + w.Eta*gradient
	w.Value -= w.Velocity
}
