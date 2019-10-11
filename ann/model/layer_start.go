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

func (s *StartLayer) GetErrorToOutput() Data {
	return NewData()
}

func (s *StartLayer) GetOutputToInput() Data {
	return s.Input.Identity2D()
}

func (s *StartLayer) GetInputSize() []uint {
	return s.Input.GetSize()
}

func (s *StartLayer) GetOutputSize() []uint {
	return s.Input.GetSize()
}
