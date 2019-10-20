package layer

import (
	. "github.com/XDean/go-machine-learning/ann/classic"
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

		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit

		Weight []Data    // K * F * F * D1
		Bias   []float64 // TODO, not used now

		InputSize Size // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		// D2 = K
		OutputSize Size // K * W2 * H2

		input          Data   // D1 * W1 * H1
		output         Data   // K * W2 * H2
		errorToOutput  Data   // output, ∂E / ∂a
		errorToWeight  []Data // weight, ∂E / ∂w
		outputPartial  Data
		outputToInput  [][][]Data // output * F * F * D1, ∂a / ∂i, no active partial
		outputToWeight [][][]Data // output * F * F * D1, ∂a / ∂w, no active partial
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
	weightSize := [3]int{f.KernelSize, f.KernelSize, inputSize[2]}
	if !f.BaseLayer.Init {
		f.BaseLayer.Init = true
		f.InputSize = inputSize
		f.OutputSize = [3]int{
			f.KernelCount,
			(inputSize[0]+2*f.Padding-f.KernelSize)/f.Stride + 1,
			(inputSize[1]+2*f.Padding-f.KernelSize)/f.Stride + 1,
		}
		f.Weight = make([]Data, f.KernelCount)
		for i := range f.Weight {
			f.Weight[i] = f.WeightInit.Init(NewData(weightSize))
		}
		f.Bias = make([]float64, f.KernelCount)
	}
	f.output = NewData(f.OutputSize)
	f.errorToOutput = NewData(f.OutputSize)
	f.errorToWeight = make([]Data, f.KernelCount)
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
	f.outputToWeight = newO2W()
	f.outputToInput = newO2W()
	f.outputPartial = NewData(f.OutputSize)
}

func (f *Convolution) Forward() {
	f.input = f.GetPrev().GetOutput()

	f.output.ForEachIndex(func(kernel, x, y int, value float64) {
		net := 0.0
		for i := 0; i < f.KernelSize; i++ {
			for j := 0; j < f.KernelSize; j++ {
				for z := 0; z < f.InputSize[0]; z++ {
					inputX := x + i - f.Padding
					inputY := y + j - f.Padding
					weight := f.Weight[kernel].Value[i][j][z]
					isPadding := inputX < 0 || inputX >= f.InputSize[1] || inputY < 0 || inputY >= f.InputSize[2]
					inputValue := 0.0
					if !isPadding {
						inputValue = f.input.Value[inputX][inputY][z]
					}
					net += inputValue * weight
					if !isPadding {
						f.outputToInput[kernel][x][y].Value[i][j][z] = weight
					}
					f.outputToWeight[kernel][x][y].Value[i][j][z] = inputValue
				}
			}
		}
		output, partial := f.Activation.Active(net)
		f.output.Value[kernel][x][y] = output
		f.outputPartial.Value[kernel][x][y] = partial
	})
}

func (f *Convolution) Backward() {
	f.errorToOutput = f.GetNext().GetErrorToInput()
	for n, v := range f.errorToWeight {
		v.MapIndex(func(i, j, k int, value float64) float64 {
			sum := 0.0
			for x := range f.errorToOutput.Value[n] {
				for y := range f.errorToOutput.Value[n][x] {
					sum += f.errorToOutput.Value[n][x][y] * f.outputToWeight[n][x][y].Value[i][j][k] * f.outputPartial.Value[n][x][y]
				}
			}
			return sum
		})
	}
}

func (f *Convolution) Learn() {
	for n, v := range f.Weight {
		v.MapIndex(func(i, j, k int, value float64) float64 {
			return value - f.LearningRatio*f.errorToWeight[n].Value[i][j][k]
		})
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
					result.Value[z][inputX][inputY] += v * f.outputToInput[kernel][x][y].Value[i][j][z] * f.outputPartial.Value[kernel][x][y]
				}
			}
		}
	})
	return result
}
