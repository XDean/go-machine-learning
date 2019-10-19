package model

type StartLayer struct {
	BaseLayer
	Input Data
}

func NewStartLayer(input Data) *StartLayer {
	return &StartLayer{Input: input}
}

func (s *StartLayer) Init() {
	// do nothing
}

func (s *StartLayer) Forward() {
	// do nothing
}

func (s *StartLayer) Backward() {
	// do nothing
}

func (s *StartLayer) Learn() {
	// do nothing
}

func (s *StartLayer) GetOutput() Data {
	return s.Input
}

func (s *StartLayer) GetErrorToInput() Data {
	return EMPTY_DATA
}

func (s *StartLayer) GetOutputSize() Size {
	return s.Input.Size
}

func (s *StartLayer) SetPrev(l Layer) {
	panic("start layer must has no prev")
}
