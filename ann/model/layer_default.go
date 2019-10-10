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

func (s *StartLayer) Backward(error float64, target Data) {
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
	Input      Data
	TotalError float64 // E
	ErrorFunc  ErrorFunc
	Error      Data // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc}
}

func (e *EndLayer) Forward() {
	e.Input = e.Prev.GetOutput()
}

func (e *EndLayer) Backward(error float64, target Data) {
	panic("implement me")
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
	return e.Error
}

func (e *EndLayer) GetInputSize() []uint {
	return e.Input.GetSize()
}

func (e *EndLayer) GetOutputSize() []uint {
	return e.Input.GetSize()
}
