package layer

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/XDean/go-machine-learning/ann/weight"
)

func init() {
	persistent.Register(new(FullConnect))
}

type (
	FullConnect struct {
		Size          int
		Activation    activation.Activation
		WeightInit    weight.Init
		WeightFactory weight.Factory

		InputSize core.Size
		Weight    [][][][]weight.Weight //  a * i
		Bias      weight.Weight
	}

	fullConnectContext struct {
		layer          *FullConnect
		output         core.Data   // a
		errorToOutput  core.Data   // a, ∂E / ∂a
		outputToInput  []core.Data // a * i, ∂a / ∂i
		outputToWeight []core.Data // a * i, ∂a / ∂w
		outputToNet    []float64   // a
	}

	FullConnectConfig struct {
		Size          int
		Activation    activation.Activation
		WeightInit    weight.Init
		WeightFactory weight.Factory
	}
)

var (
	FullConnectDefaultConfig = FullConnectConfig{
		Activation:    activation.Sigmoid{},
		WeightInit:    &weight.RandomInit{Range: 1},
		WeightFactory: weight.SGDFactory{Eta: 0.1},
	}
)

func NewFullConnect(config FullConnectConfig) *FullConnect {
	if config.Size == 0 {
		panic("Full Connect size not specified")
	}
	if config.Activation == nil {
		config.Activation = FullConnectDefaultConfig.Activation
	}
	if config.WeightFactory == nil {
		config.WeightFactory = FullConnectDefaultConfig.WeightFactory
	}
	if config.WeightInit == nil {
		config.WeightInit = FullConnectDefaultConfig.WeightInit
	}
	return &FullConnect{
		Size:          config.Size,
		Activation:    config.Activation,
		WeightInit:    config.WeightInit,
		WeightFactory: config.WeightFactory,
	}
}

func (f *FullConnect) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	f.InputSize = inputSize
	f.Weight = make([][][][]weight.Weight, f.Size)
	for i := range f.Weight {
		f.Weight[i] = weight.Create3D(f.WeightFactory, f.WeightInit, inputSize)
	}
	f.Bias = weight.Create(f.WeightFactory, f.WeightInit)
}

func (f *FullConnect) Learn(ctxs []core.Context) {
	size := float64(len(ctxs))
	for outputIndex := 0; outputIndex < f.Size; outputIndex++ {
		for i := range f.Weight[outputIndex] {
			for j := range f.Weight[outputIndex][i] {
				for k := range f.Weight[outputIndex][i][j] {
					gradient := 0.0
					for _, v := range ctxs {
						ctx := v.(*fullConnectContext)
						gradient += ctx.errorToOutput.Value[0][0][outputIndex] * ctx.outputToWeight[outputIndex].Value[i][j][k] / size
					}
					f.Weight[outputIndex][i][j][k].Learn(gradient)
				}
			}
		}
	}
	biasGradient := 0.0 // ∂E / ∂b
	for _, v := range ctxs {
		ctx := v.(*fullConnectContext)
		for outputIndex, v := range ctx.outputToNet {
			biasGradient += ctx.errorToOutput.Value[0][0][outputIndex] * v / size
		}
	}
	f.Bias.Learn(biasGradient)
}

func (f *FullConnect) NewContext() core.Context {
	return &fullConnectContext{
		layer:          f,
		output:         core.NewData([3]int{1, 1, f.Size}),
		outputToInput:  f.newOutputToInputArray(f.InputSize),
		outputToWeight: f.newOutputToInputArray(f.InputSize),
		outputToNet:    make([]float64, f.Size),
	}
}

func (f *FullConnect) newOutputToInputArray(inputSize core.Size) []core.Data {
	result := make([]core.Data, f.Size)
	for i := range result {
		result[i] = core.NewData(inputSize)
	}
	return result
}

func (f *FullConnect) GetOutputSize() core.Size {
	return core.Size{1, 1, f.Size}
}

func (f *fullConnectContext) Forward(prev core.Context) {
	input := prev.GetOutput()
	for outputIndex := 0; outputIndex < f.layer.Size; outputIndex++ {
		net := f.layer.Bias.Get()
		for i := range input.Value {
			for j := range input.Value[i] {
				for k, inputValue := range input.Value[i][j] {
					w := f.layer.Weight[outputIndex][i][j][k]
					net += w.Get() * inputValue
				}
			}
		}
		output, partial := f.layer.Activation.Active(net)
		for i := range input.Value {
			for j := range input.Value[i] {
				for k, inputValue := range input.Value[i][j] {
					w := f.layer.Weight[outputIndex][i][j][k].Get()
					f.outputToInput[outputIndex].Value[i][j][k] = w * partial
					f.outputToWeight[outputIndex].Value[i][j][k] = inputValue * partial
					f.outputToNet[outputIndex] = partial
				}
			}
		}
		f.output.Value[0][0][outputIndex] = output
	}
}

func (f *fullConnectContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *fullConnectContext) GetOutput() core.Data {
	return f.output
}

func (f *fullConnectContext) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.InputSize)
	for i := range result.Value {
		for j := range result.Value[i] {
			for k := range result.Value[i][j] {
				sum := 0.0
				for outputIndex := 0; outputIndex < f.layer.Size; outputIndex++ {
					sum += f.errorToOutput.Value[0][0][outputIndex] * f.outputToInput[outputIndex].Value[i][j][k]
				}
				result.Value[i][j][k] = sum
			}
		}
	}
	return result
}
