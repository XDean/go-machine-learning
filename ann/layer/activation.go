package layer

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(new(Activation))
}

type (
	Activation struct {
		Activation activation.Activation

		InputSize core.Size
	}

	activationContext struct {
		layer         *Activation
		output        core.Data
		errorToOutput core.Data
		outputToInput core.Data
	}
)

func NewActivation(activation activation.Activation) *Activation {
	return &Activation{Activation: activation}
}

func (f *Activation) Init(prev, next core.Layer) {
	f.InputSize = prev.GetOutputSize()
}

func (f *Activation) Learn([]core.Context) {
	// do nothing
}

func (f *Activation) NewContext() core.Context {
	return &activationContext{
		layer:         f,
		output:        core.NewData(f.InputSize),
		errorToOutput: core.NewData(f.InputSize),
		outputToInput: core.NewData(f.InputSize),
	}
}

func (f *activationContext) Forward(prev core.Context) {
	input := prev.GetOutput()
	f.output.MapIndex(func(i, j, k int, _ float64) float64 {
		output, partial := f.layer.Activation.Active(input.Value[i][j][k])
		f.outputToInput.Value[i][j][k] = partial
		return output
	})
}

func (f *activationContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *activationContext) GetOutput() core.Data {
	return f.output
}

func (f *activationContext) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.InputSize)
	result.MapIndex(func(i, j, k int, _ float64) float64 {
		return f.errorToOutput.Value[i][j][k] * f.outputToInput.Value[i][j][k]
	})
	return result
}

func (f *Activation) GetOutputSize() core.Size {
	return f.InputSize
}
