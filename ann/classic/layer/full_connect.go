package layer

import (
	. "github.com/XDean/go-machine-learning/ann/classic"
	"github.com/XDean/go-machine-learning/ann/core/data"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(new(FullConnect))
}

type (
	FullConnect struct {
		BaseLayer

		Weight data.Data // a * i
		Bias   float64   // TODO, not used now

		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit

		input          data.Data // i
		output         data.Data // a
		errorToOutput  data.Data // a, ∂E / ∂a
		errorToInput   data.Data // i, ∂E / ∂a
		outputToInput  data.Data // a * i, ∂a / ∂i
		outputToWeight data.Data // a * i, ∂a / ∂w
	}

	FullConnectConfig struct {
		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}
)

var (
	FullConnectDefaultConfig = FullConnectConfig{
		Size:          10,
		Activation:    Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &RandomInit{Range: 1},
	}
)

func NewFullConnect(config FullConnectConfig) *FullConnect {
	if config.Size == 0 {
		config.Size = FullConnectDefaultConfig.Size
	}
	if config.Activation == nil {
		config.Activation = FullConnectDefaultConfig.Activation
	}
	if config.LearningRatio == 0 {
		config.LearningRatio = FullConnectDefaultConfig.LearningRatio
	}
	if config.WeightInit == nil {
		config.WeightInit = FullConnectDefaultConfig.WeightInit
	}
	return &FullConnect{
		Size:          config.Size,
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *FullConnect) Init() {
	inputSize := f.GetPrev().GetOutputSize()
	if f.Weight == nil {
		f.Weight = f.WeightInit.Init(data.NewData(append([]int{f.Size}, inputSize...)))
	}
	f.output = data.NewData1(f.Size)
	f.outputToInput = data.NewData(append([]int{f.Size}, inputSize...))
	f.outputToWeight = data.NewData(append([]int{f.Size}, inputSize...))
}

func (f *FullConnect) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output.MapIndex(func(outputIndex []int, _ float64) float64 {
		net := 0.0
		f.Weight.GetData(outputIndex).ForEachIndex(func(inputIndex []int, w float64) {
			input := f.input.GetValue(inputIndex)
			net += w * input
			index := append(outputIndex, inputIndex...)
			f.outputToInput.SetValue(w, index)
			f.outputToWeight.SetValue(input, index)
		})
		output, partial := f.Activation.Active(net)
		f.outputToInput.GetData(outputIndex).Map(func(value float64) float64 { return value * partial })
		f.outputToWeight.GetData(outputIndex).Map(func(value float64) float64 { return value * partial })
		return output
	})
}

func (f *FullConnect) Backward() {
	f.errorToOutput = f.GetNext().GetErrorToInput()
	f.errorToInput = ErrorToInput(f.errorToOutput, f.outputToInput)
}

func (f *FullConnect) Learn() {
	f.Weight.MapIndex(func(index []int, value float64) float64 {
		return value - f.LearningRatio*f.errorToOutput.GetValue([]int{index[0]})*f.outputToWeight.GetValue(index)
	})
}

func (f *FullConnect) GetInput() data.Data {
	return f.input
}

func (f *FullConnect) GetOutput() data.Data {
	return f.output
}

func (f *FullConnect) GetErrorToInput() data.Data {
	return f.errorToInput
}

func (f *FullConnect) GetOutputSize() []int {
	return []int{f.Size}
}
