package model

import "encoding/gob"

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

func (e *EndLayer) Name() string {
	return "End Layer"
}

func (e *EndLayer) Save(writer *gob.Encoder) error {
	panic("no save")
}

func (e *EndLayer) Load(reader *gob.Decoder) error {
	panic("no load")
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

func (e *EndLayer) ToResult() Result {
	return Result{
		Output:     e.Input,
		Target:     e.Target,
		TotalError: e.TotalError,
	}
}

func (e *EndLayer) SetNext(l Layer) {
	panic("last layer must have no next")
}