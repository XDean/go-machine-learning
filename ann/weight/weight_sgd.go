package weight

type SGD struct {
	Value float64
	Eta   float64
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
