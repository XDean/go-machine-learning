package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/XDean/go-machine-learning/ann/weight"
)

func init() {
	persistent.Register(new(SubSampling))
}

type (
	SubSampling struct {
		Size          int // F
		Stride        int // S
		Padding       int // P
		WeightFactory weight.Factory
		WeightInit    weight.Init

		InputSize core.Size // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		OutputSize core.Size // D1 * W2 * H2

		Weight []weight.Weight // D1
		Bias   []weight.Weight // D1
	}

	subSamplingContext struct {
		layer          *SubSampling
		input          core.Data
		output         core.Data
		errorToOutput  core.Data     // output
		errorToInput   core.Data     // input
		outputToWeight core.Data     // output
		outputToInput  [][]core.Data // F * F * output
	}

	SubSamplingConfig struct {
		Size    int
		Padding int
		Stride  int
	}
)

var (
	SubSamplingDefaultConfig = SubSamplingConfig{
		Size:    2,
		Padding: 0,
		Stride:  2,
	}
)

func NewSubSampling(config SubSamplingConfig) *SubSampling {
	if config.Size == 0 {
		config.Size = SubSamplingDefaultConfig.Size
	}
	if config.Stride == 0 {
		config.Stride = SubSamplingDefaultConfig.Stride
	}
	return &SubSampling{
		Size:    config.Size,
		Stride:  config.Stride,
		Padding: config.Padding,
	}
}

func (f *SubSampling) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	f.InputSize = inputSize
	f.OutputSize = [3]int{
		f.InputSize[0],
		(f.InputSize[1]+2*f.Padding-f.Size)/f.Stride + 1,
		(f.InputSize[2]+2*f.Padding-f.Size)/f.Stride + 1,
	}
	f.Weight = weight.Create1D(f.WeightFactory, f.WeightInit, f.InputSize[0])
	f.Bias = weight.Create1D(f.WeightFactory, f.WeightInit, f.InputSize[0])
}

func (f *SubSampling) Learn(ctxs []core.Context) {
	for i := range f.Weight {
		weightGradient := 0.0
		for _, v := range ctxs {
			ctx := v.(*subSamplingContext)
			for x := range ctx.errorToOutput.Value[i] {
				for y := range ctx.errorToOutput.Value[i][x] {
					weightGradient += ctx.errorToOutput.Value[i][x][y] * ctx.outputToWeight.Value[i][x][y]
				}
			}
		}
		f.Weight[i].Learn(weightGradient)
	}
	for i := range f.Bias {
		biasGradient := 0.0
		for _, v := range ctxs {
			ctx := v.(*subSamplingContext)
			for x := range ctx.errorToOutput.Value[i] {
				for y := range ctx.errorToOutput.Value[i][x] {
					biasGradient += ctx.errorToOutput.Value[i][x][y]
				}
			}
		}
		f.Bias[i].Learn(biasGradient)
	}
}

func (f *SubSampling) NewContext() core.Context {
	outputToInput := make([][]core.Data, f.Size)
	for i := range outputToInput {
		outputToInput[i] = make([]core.Data, f.Size)
		for j := range outputToInput[i] {
			outputToInput[i][j] = core.NewData(f.OutputSize)
		}
	}
	return &subSamplingContext{
		layer:          f,
		output:         core.NewData(f.OutputSize),
		errorToOutput:  core.NewData(f.OutputSize),
		outputToInput:  outputToInput,
		outputToWeight: core.NewData(f.OutputSize),
	}
}

func (f *subSamplingContext) Forward(prev core.Context) {
	f.input = prev.GetOutput()
	f.output.MapIndex(func(dep, x, y int, _ float64) float64 {
		sum := 0.0
		for i := 0; i < f.layer.Size; i++ {
			for j := 0; j < f.layer.Size; j++ {
				inputX := x + i - f.layer.Padding
				inputY := y + j - f.layer.Padding
				isPadding := inputX < 0 || inputX >= f.layer.InputSize[0] || inputY < 0 || inputY >= f.layer.InputSize[1]
				inputValue := 0.0
				if !isPadding {
					inputValue = f.input.Value[dep][inputX][inputY]
				}
				sum += inputValue
			}
		}
		f.outputToWeight.Value[dep][x][y] = sum
		return sum*f.layer.Weight[dep].Get() + f.layer.Bias[dep].Get()
	})
}

func (f *subSamplingContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *subSamplingContext) GetOutput() core.Data {
	return f.output
}

func (f *subSamplingContext) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.InputSize)
	f.errorToOutput.ForEachIndex(func(dep, x, y int, value float64) {
		for i := 0; i < f.layer.Size; i++ {
			for j := 0; j < f.layer.Size; j++ {
				inputX := x + i - f.layer.Padding
				inputY := y + j - f.layer.Padding
				isPadding := inputX < 0 || inputX >= f.layer.InputSize[0] || inputY < 0 || inputY >= f.layer.InputSize[1]
				if isPadding {
					continue
				}
				result.Value[dep][inputX][inputY] += value
			}
		}
	})
	return result
}

func (f *SubSampling) GetOutputSize() core.Size {
	return f.OutputSize
}

func (f *SubSampling) Desc() core.Desc {
	return core.SimpleDesc{
		Name: "SubSampling",
		Core: f.OutputSize,
		Params: map[string]interface{}{
			"Size":    f.Size,
			"Stride":  f.Stride,
			"Padding": f.Padding,
			"Weight":  f.WeightFactory,
		},
	}
}
