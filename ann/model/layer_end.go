package model

type EndLayer struct {
	BaseLayer

	ErrorFunc ErrorFunc
	Target    Data

	Input         Data
	TotalError    float64 // E
	ErrorToOutput Data    // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc, target Data) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc, Target: target}
}

func (e *EndLayer) Forward() {
	e.Input = e.Prev.GetOutput()
	e.TotalError, e.ErrorToOutput = e.ErrorFunc(e.Target, e.Input)
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

func (e *EndLayer) GetErrorToOutput() Data {
	return e.ErrorToOutput
}

func (e *EndLayer) GetOutputToInput() Data {
	return e.Input.Identity2D()
}

func (e *EndLayer) GetInputSize() []uint {
	return e.Input.GetSize()
}

func (e *EndLayer) GetOutputSize() []uint {
	return e.Input.GetSize()
}
