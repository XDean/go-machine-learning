package model

type StartLayer struct {
	BaseLayer
	Input Data
}

func NewStartLayer(input Data) *StartLayer {
	return &StartLayer{Input: input}
}

func (s *StartLayer) Forward() {
	s.Next.Forward()
}

func (s *StartLayer) Backward() {
	// do nothing
}

func (s *StartLayer) Learn() {
	// do nothing
}

func (s *StartLayer) GetInput() Data {
	return s.Input
}

func (s *StartLayer) GetOutput() Data {
	return s.Input
}

func (s *StartLayer) GetWeight() Data {
	return NewData()
}

func (s *StartLayer) GetError() Data {
	return NewData()
}

func (s *StartLayer) GetInputSize() []uint {
	return s.Input.GetSize()
}

func (s *StartLayer) GetOutputSize() []uint {
	return s.Input.GetSize()
}

type EndLayer struct {
	BaseLayer

	ErrorFunc ErrorFunc
	Target    Data

	Input        Data
	TotalError   float64 // E
	PartialError Data    // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc, target Data) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc, Target: target}
}

func (e *EndLayer) Forward() {
	e.Input = e.Prev.GetOutput()
	e.TotalError, e.PartialError = e.ErrorFunc(e.Target, e.Input)
}

func (e *EndLayer) Backward() {
	e.Prev.Backward()
}

func (e *EndLayer) Learn() {
	// do nothing
}

func (e *EndLayer) GetInput() Data {
	return e.Input
}

func (e *EndLayer) GetOutput() Data {
	return e.Input
}

func (e *EndLayer) GetWeight() Data {
	return e.Input
}

func (e *EndLayer) GetError() Data {
	return e.PartialError
}

func (e *EndLayer) GetInputSize() []uint {
	return e.Input.GetSize()
}

func (e *EndLayer) GetOutputSize() []uint {
	return e.Input.GetSize()
}
