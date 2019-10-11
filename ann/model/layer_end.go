package model

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
)

type EndLayer struct {
	BaseLayer

	ErrorFunc ErrorFunc
	Target    base.Data

	Input         base.Data
	TotalError    float64   // E
	ErrorToOutput base.Data // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc, target base.Data) *EndLayer {
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
	e.TotalError, e.ErrorToOutput = e.ErrorFunc.CalcError(e.Target, e.Input)
}

func (e *EndLayer) Backward() {
	e.Prev.Backward()
}

func (e *EndLayer) Learn() {
	// do nothing
}

func (e *EndLayer) GetInput() base.Data {
	return e.Input
}

func (e *EndLayer) GetOutput() base.Data {
	return e.Input
}

func (e *EndLayer) GetErrorToOutput() base.Data {
	return e.ErrorToOutput
}

func (e *EndLayer) GetOutputToInput() base.Data {
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
