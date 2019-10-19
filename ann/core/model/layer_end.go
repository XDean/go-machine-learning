package model

import (
	"time"
)

type EndLayer struct {
	BaseLayer

	ErrorFunc ErrorFunc
	Target    Data

	Input        Data
	TotalError   float64 // E
	ErrorToInput Data    // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc, target Data) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc, Target: target}
}

func (e *EndLayer) Name() string {
	return "End Layer"
}

func (e *EndLayer) Init() {
	// do nothing
}

func (e *EndLayer) Forward() {
	e.Input = e.prev.GetOutput()
	e.TotalError, e.ErrorToInput = e.ErrorFunc.CalcError(e.Target, e.Input)
}

func (e *EndLayer) Backward() {
	// do nothing
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

func (e *EndLayer) GetErrorToInput() Data {
	return e.ErrorToInput
}

func (e *EndLayer) GetOutputSize() Size {
	return e.Input.Size
}

func (e *EndLayer) ToResult(start time.Time) Result {
	return Result{
		Output:     e.Input,
		Target:     e.Target,
		TotalError: e.TotalError,
		Time:       time.Since(start),
	}
}

func (e *EndLayer) SetNext(l Layer) {
	panic("last layer must have no next")
}
