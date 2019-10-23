package layer

import (
	"github.com/XDean/go-machine-learning/ann/classic/activation"
	"github.com/XDean/go-machine-learning/ann/classic/weight"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
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

		Activation    activation.Activation
		LearningRatio float64
		WeightInit    weight.Init

		Weight []Data    // K * F * F * D1
		Bias   []float64 // K

		InputSize Size // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		// D2 = K
		OutputSize Size // K * W2 * H2

		input         Data   // D1 * W1 * H1
		output        Data   // K * W2 * H2
		errorToOutput Data   // output, ∂E / ∂a
		errorToWeight []Data // weight, ∂E / ∂w
		errorToBias   []float64
		outputToNet   Data
		netToInput    [][][]Data // output * F * F * D1, ∂a / ∂i, no active partial
		netToWeight   [][][]Data // output * F * F * D1, ∂a / ∂w, no active partial
	}

	ConvolutionConfig struct {
		KernelCount   int // K
		KernelSize    int // F
		Stride        int // S
		Padding       int // P
		Activation    activation.Activation
		LearningRatio float64
		WeightInit    weight.Init
	}
)

var (
	ConvolutionDefaultConfig = ConvolutionConfig{
		KernelCount:   1,
		KernelSize:    3,
		Stride:        1,
		Activation:    activation.Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &weight.NormalInit{Mean: 0, Std: 0.1},
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
		KernelCount:   config.KernelCount,
		KernelSize:    config.KernelSize,
		Stride:        config.Stride,
		Padding:       config.Padding,
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *Convolution) Init() {
	inputSize := f.GetPrev().GetOutputSize()
	weightSize := [3]int{inputSize[0], f.KernelSize, f.KernelSize}
	if !f.BaseLayer.Init {
		f.BaseLayer.Init = true
		f.InputSize = inputSize
		f.OutputSize = [3]int{
			f.KernelCount,
			(inputSize[1]+2*f.Padding-f.KernelSize)/f.Stride + 1,
			(inputSize[2]+2*f.Padding-f.KernelSize)/f.Stride + 1,
		}
		f.Weight = make([]Data, f.KernelCount)
		for i := range f.Weight {
			f.Weight[i] = NewData(weightSize)
			f.WeightInit.InitData(f.Weight[i])
		}
		f.Bias = make([]float64, f.KernelCount)
	}
	f.output = NewData(f.OutputSize)
	f.errorToOutput = NewData(f.OutputSize)
	f.errorToWeight = make([]Data, f.KernelCount)
	f.errorToBias = make([]float64, f.KernelCount)
	for i := range f.errorToWeight {
		f.errorToWeight[i] = NewData(weightSize)
	}
	newO2W := func() [][][]Data {
		result := make([][][]Data, f.OutputSize[0])
		for i := range result {
			result[i] = make([][]Data, f.OutputSize[1])
			for j := range result[i] {
				result[i][j] = make([]Data, f.OutputSize[2])
				for k := range result[i][j] {
					result[i][j][k] = NewData(f.Weight[0].Size)
				}
			}
		}
		return result
	}
	f.netToWeight = newO2W()
	f.netToInput = newO2W()
	f.outputToNet = NewData(f.OutputSize)
}

func (f *Convolution) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output.ForEachIndex(func(kernel, x, y int, value float64) {
		net := f.Bias[kernel]
		for i := 0; i < f.KernelSize; i++ {
			for j := 0; j < f.KernelSize; j++ {
				for z := 0; z < f.InputSize[0]; z++ {
					inputX := x + i - f.Padding
					inputY := y + j - f.Padding
					weight := f.Weight[kernel].Value[z][i][j]
					isPadding := inputX < 0 || inputX >= f.InputSize[1] || inputY < 0 || inputY >= f.InputSize[2]
					inputValue := 0.0
					if !isPadding {
						inputValue = f.input.Value[z][inputX][inputY]
					}
					net += inputValue * weight
					if !isPadding {
						f.netToInput[kernel][x][y].Value[z][i][j] = weight
						f.netToWeight[kernel][x][y].Value[z][i][j] = inputValue
					}
				}
			}
		}
		output, partial := f.Activation.Active(net)
		f.output.Value[kernel][x][y] = output
		f.outputToNet.Value[kernel][x][y] = partial
	})
}

func (f *Convolution) Backward() {
	f.errorToOutput = f.GetNext().GetErrorToInput()
	for kernel, v := range f.errorToWeight {
		v.MapIndex(func(i, j, k int, value float64) float64 {
			sum := 0.0
			for x := range f.errorToOutput.Value[kernel] {
				for y := range f.errorToOutput.Value[kernel][x] {
					sum += f.errorToOutput.Value[kernel][x][y] * f.outputToNet.Value[kernel][x][y] * f.netToWeight[kernel][x][y].Value[i][j][k]
				}
			}
			return sum
		})
		sum := 0.0
		for x := range f.errorToOutput.Value[kernel] {
			for y := range f.errorToOutput.Value[kernel][x] {
				sum += f.errorToOutput.Value[kernel][x][y] * f.outputToNet.Value[kernel][x][y]
			}
		}
		f.errorToBias[kernel] = sum
	}
}

func (f *Convolution) Learn() {
	for n, v := range f.Weight {
		v.MapIndex(func(i, j, k int, value float64) float64 {
			return value - f.LearningRatio*f.errorToWeight[n].Value[i][j][k]
		})
	}
	for i := range f.Bias {
		f.Bias[i] -= f.LearningRatio * f.errorToBias[i]
	}
}

func (f *Convolution) GetInput() Data {
	return f.input
}

func (f *Convolution) GetOutput() Data {
	return f.output
}

func (f *Convolution) GetOutputSize() Size {
	return f.OutputSize
}

func (f *Convolution) GetErrorToInput() Data {
	result := NewData(f.InputSize)
	f.errorToOutput.ForEachIndex(func(kernel, x, y int, v float64) {
		for i := 0; i < f.KernelSize; i++ {
			for j := 0; j < f.KernelSize; j++ {
				for z := 0; z < f.InputSize[0]; z++ {
					inputX := x + i - f.Padding
					inputY := y + j - f.Padding
					isPadding := inputX < 0 || inputX >= f.InputSize[1] || inputY < 0 || inputY >= f.InputSize[2]
					if isPadding {
						continue
					}
					result.Value[z][inputX][inputY] += v * f.netToInput[kernel][x][y].Value[z][i][j] * f.outputToNet.Value[kernel][x][y]
				}
			}
		}
	})
	return result
}
