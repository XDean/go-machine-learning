package layer

import (
	"github.com/XDean/go-machine-learning/ann/classic/activation"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(new(Activation))
}

type (
	Activation struct {
		Activation activation.Activation

		InputSize Size
	}

	activationContext struct {
		layer         *Activation
		output        Data
		errorToOutput Data
		outputToInput Data
	}
)

func NewActivation(activation activation.Activation) *Activation {
	return &Activation{Activation: activation}
}

func (f *Activation) Init(prev, next Layer) {
	f.InputSize = prev.GetOutputSize()
}

func (f *Activation) Learn([]Context) {
	// do nothing
}

func (f *Activation) NewContext() Context {
	return &activationContext{
		output:        NewData(f.InputSize),
		errorToOutput: NewData(f.InputSize),
		outputToInput: NewData(f.InputSize),
	}
}

func (f *activationContext) Forward(prev Context) {
	input := prev.GetOutput()
	f.output.MapIndex(func(i, j, k int, _ float64) float64 {
		output, partial := f.layer.Activation.Active(input.Value[i][j][k])
		f.outputToInput.Value[i][j][k] = partial
		return output
	})
}

func (f *activationContext) Backward(next Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *activationContext) GetOutput() Data {
	return f.output
}

func (f *activationContext) GetErrorToInput() Data {
	result := NewData(f.layer.InputSize)
	result.MapIndex(func(i, j, k int, _ float64) float64 {
		return f.errorToOutput.Value[i][j][k] * f.outputToInput.Value[i][j][k]
	})
	return result
}

func (f *Activation) GetOutputSize() Size {
	return f.InputSize
}
