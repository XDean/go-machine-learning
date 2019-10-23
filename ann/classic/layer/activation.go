package layer

import (
	"github.com/XDean/go-machine-learning/ann/classic/activation"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(new(Activation))
}

type Activation struct {
	BaseLayer

	Activation activation.Activation

	InputSize Size

	output        Data
	errorToOutput Data
	outputToInput Data
}

func NewActivation(activation activation.Activation) *Activation {
	return &Activation{Activation: activation}
}

func (f *Activation) Init() {
	inputSize := f.GetPrev().GetOutputSize()
	if !f.BaseLayer.Init {
		f.BaseLayer.Init = true
		f.InputSize = inputSize
	}
	f.output = NewData(inputSize)
	f.errorToOutput = NewData(inputSize)
	f.outputToInput = NewData(inputSize)
}

func (f *Activation) Forward() {
	input := f.GetPrev().GetOutput()
	f.output.MapIndex(func(i, j, k int, _ float64) float64 {
		output, partial := f.Activation.Active(input.Value[i][j][k])
		f.outputToInput.Value[i][j][k] = partial
		return output
	})
}

func (f *Activation) Backward() {
	f.errorToOutput = f.GetNext().GetErrorToInput()
}

func (f *Activation) Learn() {
	// do nothing
}

func (f *Activation) GetOutput() Data {
	return f.output
}

func (f *Activation) GetErrorToInput() Data {
	result := NewData(f.InputSize)
	result.MapIndex(func(i, j, k int, _ float64) float64 {
		return f.errorToOutput.Value[i][j][k] * f.outputToInput.Value[i][j][k]
	})
	return result
}

func (f *Activation) GetOutputSize() Size {
	return f.InputSize
}
