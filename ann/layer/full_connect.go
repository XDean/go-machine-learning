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
		LearningRatio float64
		WeightInit    weight.Init

		InputSize core.Size
		Weight    []core.Data //  a * i
		Bias      float64
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
		LearningRatio float64
		WeightInit    weight.Init
	}
)

var (
	FullConnectDefaultConfig = FullConnectConfig{
		Activation:    activation.Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &weight.RandomInit{Range: 1},
	}
)

func NewFullConnect(config FullConnectConfig) *FullConnect {
	if config.Size == 0 {
		panic("Full Connect size not specified")
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

func (f *FullConnect) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	f.InputSize = inputSize
	f.Weight = f.newOutputToInputArray(inputSize)
	for _, v := range f.Weight {
		f.WeightInit.InitData(v)
	}
	f.Bias = f.WeightInit.InitOne()
}

func (f *FullConnect) Learn(ctxs []core.Context) {
	size := float64(len(ctxs))
	for _, v := range ctxs {
		ctx := v.(*fullConnectContext)
		for outputIndex := 0; outputIndex < f.Size; outputIndex++ {
			for i := range f.Weight[outputIndex].Value {
				for j := range f.Weight[outputIndex].Value[i] {
					for k := range f.Weight[outputIndex].Value[i][j] {
						f.Weight[outputIndex].Value[i][j][k] -=
							f.LearningRatio * ctx.errorToOutput.Value[0][0][outputIndex] * ctx.outputToWeight[outputIndex].Value[i][j][k] / size
					}
				}
			}
		}
		biasPartial := 0.0 // ∂E / ∂b
		for outputIndex, v := range ctx.outputToNet {
			biasPartial += ctx.errorToOutput.Value[0][0][outputIndex] * v
		}
		f.Bias -= f.LearningRatio * biasPartial / size
	}
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
		net := f.layer.Bias
		for i := range input.Value {
			for j := range input.Value[i] {
				for k, inputValue := range input.Value[i][j] {
					w := f.layer.Weight[outputIndex].Value[i][j][k]
					net += w * inputValue
				}
			}
		}
		output, partial := f.layer.Activation.Active(net)
		for i := range input.Value {
			for j := range input.Value[i] {
				for k, inputValue := range input.Value[i][j] {
					w := f.layer.Weight[outputIndex].Value[i][j][k]
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
