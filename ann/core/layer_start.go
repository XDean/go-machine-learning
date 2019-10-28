package core

type (
	StartLayer struct {
		Input Data
	}
)

func NewStartLayer(input Data) *StartLayer {
	return &StartLayer{Input: input}
}

func (s *StartLayer) Init(prev, next Layer) {
	//do nothing
}

func (s *StartLayer) Learn(ctxs []Context) {
	//do nothing
}

func (s *StartLayer) Forward(prev Context) {
	//do nothing
}

func (s *StartLayer) Backward(next Context) {
	//do nothing
}

func (s *StartLayer) NewContext() Context {
	return s
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
