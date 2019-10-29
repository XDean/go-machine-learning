package weight

import "github.com/XDean/go-machine-learning/ann/persistent"

func init() {
	persistent.Register(&SGD{})
	persistent.Register(SGDFactory{})
}

type (
	SGD struct {
		Value float64
		Eta   float64
	}

	SGDFactory struct {
		Eta float64
	}
)

func (f SGDFactory) Create() Weight {
	return &SGD{
		Value: 0,
		Eta:   f.Eta,
	}
}

func (s *SGD) Get() float64 {
	return s.Value
}

func (s *SGD) Set(value float64) {
	s.Value = value
}

func (s *SGD) Learn(gradient float64) {
	s.Value -= gradient * s.Eta
}
