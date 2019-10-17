package layer

import (
	. "github.com/XDean/go-machine-learning/ann/classic"
	"github.com/XDean/go-machine-learning/ann/core/data"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"github.com/XDean/go-machine-learning/ann/core/util"
)

func init() {
	persistent.Register(new(Convolution))
}

type (
	Convolution struct {
		BaseLayer

		KernelCount int // K
		KernelSize  int // F
		Stride      int // S
		Padding     int // P

		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit

		Weight data.Data // K * F * F * D1
		Bias   []float64 // TODO, not used now

		InputSize [3]int // W1 * H1 * D1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		// D2 = K
		OutputSize [3]int // W2 * H2 * K

		input          data.Data // W1 * H1 * D1
		output         data.Data // W2 * H2 * K
		errorToOutput  data.Data // output, ∂E / ∂a
		errorToWeight  data.Data // weight, ∂E / ∂w
		outputToInput  data.Data // output * input, ∂a / ∂i
		outputToWeight data.Data // output * weight, ∂a / ∂w
	}

	ConvolutionConfig struct {
		KernelCount   int // K
		KernelSize    int // F
		Stride        int // S
		Padding       int // P
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}
)

var (
	ConvolutionDefaultConfig = ConvolutionConfig{
		KernelCount:   1,
		KernelSize:    3,
		Stride:        1,
		Activation:    Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &NormalInit{Mean: 0, Std: 0.1},
	}
)

func NewConvolution(config ConvolutionConfig) *Convolution {
	if config.KernelCount == 0 {
		config.KernelCount = ConvolutionDefaultConfig.KernelCount
	}
	if config.KernelSize == 0 {
		config.KernelSize = ConvolutionDefaultConfig.KernelSize
	}
	if config.Stride == 0 {
		config.Stride = ConvolutionDefaultConfig.Stride
	}
	if config.Activation == nil {
		config.Activation = ConvolutionDefaultConfig.Activation
	}
	if config.LearningRatio == 0 {
		config.LearningRatio = ConvolutionDefaultConfig.LearningRatio
	}
	if config.WeightInit == nil {
		config.WeightInit = ConvolutionDefaultConfig.WeightInit
	}
	return &Convolution{
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *Convolution) Init() {
	util.MustTrue(f.KernelSize%2 == 1, "Kernel size must be odd")
	inputSize := f.GetPrev().GetOutputSize()
	util.MustTrue(len(inputSize) == 2 || len(inputSize) == 3)
	switch len(inputSize) {
	case 2:
		f.InputSize = [3]int{inputSize[0], inputSize[1], 1}
	case 3:
		f.InputSize = [3]int{inputSize[0], inputSize[1], inputSize[2]}
	default:
		util.MustTrue(false, "Input must be 2 or 3 dim")
	}
	f.OutputSize = [3]int{
		(f.InputSize[0]+2*f.Padding-f.KernelSize)/f.Stride + 1,
		(f.InputSize[1]+2*f.Padding-f.KernelSize)/f.Stride + 1,
		f.KernelCount,
	}

	f.Weight = f.WeightInit.Init(data.NewData(f.KernelCount, f.KernelSize, f.KernelSize, f.InputSize[2]))
	f.Bias = make([]float64, f.KernelCount)
}

func (f *Convolution) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output = data.NewData(f.OutputSize[:]...)
	f.errorToOutput = data.NewData(f.OutputSize[:]...)
	f.errorToWeight = data.NewData(f.Weight.GetSize()...)
	f.outputToInput = data.NewData(append(f.OutputSize[:], f.InputSize[:]...)...)
	f.outputToWeight = data.NewData(append(f.OutputSize[:], f.Weight.GetSize()...)...)

	halfKernelSize := (f.KernelSize - 1) / 2
	depth := f.InputSize[2]
	f.output.Map(func(outputIndex []int, _ float64) float64 {
		kernel := outputIndex[2]
		net := 0.0
		for i := -halfKernelSize; i <= halfKernelSize; i++ {
			for j := -halfKernelSize; j <= halfKernelSize; j++ {
				for z := 0; z < depth; z++ {
					weight := f.Weight.GetValue(kernel, i+halfKernelSize, j+halfKernelSize, z)
					inputValue := f.input.GetValue(outputIndex[0]+i, outputIndex[1]+j, z)
					net += inputValue * weight
					f.outputToInput.SetValue(weight, append(outputIndex, outputIndex[0]+i, outputIndex[1]+j, z)...)
					f.outputToWeight.SetValue(inputValue, append(outputIndex, i+halfKernelSize, j+halfKernelSize, z)...)
				}
			}
		}
		output, partial := f.Activation.Active(net)
		f.outputToInput.Map(func(_ []int, value float64) float64 { return value * partial })
		f.outputToWeight.Map(func(_ []int, value float64) float64 { return value * partial })
		return output
	})
}

func (f *Convolution) Backward() {
	f.errorToOutput = ErrorToInput(f.GetNext()) // next layer's input is this layer's output
	f.errorToWeight.Map(func(weightIndex []int, _ float64) float64 {
		kernel := weightIndex[0]
		sum := 0.0
		f.errorToOutput.ForEach(func(outputIndex []int, value float64) {
			if outputIndex[2] == kernel {
				sum += value * f.outputToWeight.GetValue(append(outputIndex, weightIndex...)...)
			}
		})
		return sum
	})
}

func (f *Convolution) Learn() {
	f.Weight.Map(func(index []int, value float64) float64 {
		return value - f.LearningRatio*f.errorToWeight.GetValue(index...)
	})
}

func (f *Convolution) GetInput() data.Data {
	return f.input
}

func (f *Convolution) GetOutput() data.Data {
	return f.output
}

func (f *Convolution) GetErrorToOutput() data.Data {
	return f.errorToOutput
}

func (f *Convolution) GetOutputToInput() data.Data {
	return f.outputToInput
}

func (f *Convolution) GetOutputSize() []int {
	return f.OutputSize[:]
}
