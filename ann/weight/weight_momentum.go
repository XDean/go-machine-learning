package weight

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(&Momentum{})
	persistent.Register(MomentumFactory{})
}

type (
	Momentum struct {
		Value    float64
		Eta      float64
		Gamma    float64
		Velocity float64
	}
	MomentumFactory struct {
		Eta   float64
		Gamma float64
	}
)

func DefaultMomentumFactory() MomentumFactory {
	return MomentumFactory{
		Eta:   0.1,
		Gamma: 0.5,
	}
}

func (f MomentumFactory) Desc() core.Desc {
	return core.SimpleDesc{
		Name: "Momentum",
		Params: map[string]interface{}{
			"η": f.Eta,
			"γ": f.Gamma,
		},
	}
}

func (f MomentumFactory) Create() Weight {
	return &Momentum{
		Value: 0,
		Eta:   f.Eta,
		Gamma: f.Gamma,
	}
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
