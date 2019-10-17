package model

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"time"
)

type EndLayer struct {
	BaseLayer

	ErrorFunc ErrorFunc
	Target    data.Data

	Input         data.Data
	TotalError    float64   // E
	ErrorToOutput data.Data // ∂E/∂a
}

func NewEndLayer(errorFunc ErrorFunc, target data.Data) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc, Target: target}
}

func (e *EndLayer) Name() string {
	return "End Layer"
}

func (e *EndLayer) Init() {
	// do nothing
}

func (e *EndLayer) Save(writer persistent.Encoder) error {
	panic("no save")
}

func (e *EndLayer) Load(reader persistent.Decoder) error {
	panic("no load")
}

func (e *EndLayer) Forward() {
	e.Input = e.prev.GetOutput()
	e.TotalError, e.ErrorToOutput = e.ErrorFunc.CalcError(e.Target, e.Input)
}

func (e *EndLayer) Backward() {
	// do nothing
}

func (e *EndLayer) Learn() {
	// do nothing
}

func (e *EndLayer) GetInput() data.Data {
	return e.Input
}

func (e *EndLayer) GetOutput() data.Data {
	return e.Input
}

func (e *EndLayer) GetErrorToOutput() data.Data {
	return e.ErrorToOutput
}

func (e *EndLayer) GetOutputToInput() data.Data {
	return data.Identity2D(e.Input)
}

func (e *EndLayer) SetInputSize() []int {
	return e.Input.GetSize()
}

func (e *EndLayer) GetOutputSize() []int {
	return e.Input.GetSize()
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
