package layer

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/XDean/go-machine-learning/ann/weight"
)

func init() {
	persistent.Register(new(Convolution))
}

type (
	Convolution struct {
		KernelCount int // K
		KernelSize  int // F
		Stride      int // S
		Padding     int // P

		Activation    activation.Activation
		LearningRatio float64
		WeightInit    weight.Init

		Weight []core.Data // K * F * F * D1
		Bias   []float64   // K

		InputSize core.Size // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		// D2 = K
		OutputSize core.Size // K * W2 * H2
	}

	convolutionContext struct {
		layer         *Convolution
		input         core.Data   // D1 * W1 * H1
		output        core.Data   // K * W2 * H2
		errorToOutput core.Data   // output, ∂E / ∂a
		errorToWeight []core.Data // weight, ∂E / ∂w
		errorToBias   []float64
		outputToNet   core.Data
		netToInput    [][][]core.Data // output * F * F * D1, ∂a / ∂i, no active partial
		netToWeight   [][][]core.Data // output * F * F * D1, ∂a / ∂w, no active partial
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
func (f *Convolution) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	weightSize := [3]int{inputSize[0], f.KernelSize, f.KernelSize}
	f.InputSize = inputSize
	f.OutputSize = [3]int{
		f.KernelCount,
		(inputSize[1]+2*f.Padding-f.KernelSize)/f.Stride + 1,
		(inputSize[2]+2*f.Padding-f.KernelSize)/f.Stride + 1,
	}
	f.Weight = make([]core.Data, f.KernelCount)
	for i := range f.Weight {
		f.Weight[i] = core.NewData(weightSize)
		f.WeightInit.InitData(f.Weight[i])
	}
	f.Bias = make([]float64, f.KernelCount)
}

func (f *Convolution) Learn(ctxs []core.Context) {
	size := float64(len(ctxs))
	for _, v := range ctxs {
		ctx := v.(*convolutionContext)
		for n, v := range f.Weight {
			v.MapIndex(func(i, j, k int, value float64) float64 {
				return value - f.LearningRatio*ctx.errorToWeight[n].Value[i][j][k]/size
			})
		}
		for i := range f.Bias {
			f.Bias[i] -= f.LearningRatio * ctx.errorToBias[i] / size
		}
	}
}

func (f *Convolution) NewContext() core.Context {
	errorToWeight := make([]core.Data, f.KernelCount)
	weightSize := [3]int{f.InputSize[0], f.KernelSize, f.KernelSize}
	for i := range errorToWeight {
		errorToWeight[i] = core.NewData(weightSize)
	}
	newO2W := func() [][][]core.Data {
		result := make([][][]core.Data, f.OutputSize[0])
		for i := range result {
			result[i] = make([][]core.Data, f.OutputSize[1])
			for j := range result[i] {
				result[i][j] = make([]core.Data, f.OutputSize[2])
				for k := range result[i][j] {
					result[i][j][k] = core.NewData(f.Weight[0].Size)
				}
			}
		}
		return result
	}
	return &convolutionContext{
		layer:         f,
		output:        core.NewData(f.OutputSize),
		errorToOutput: core.NewData(f.OutputSize),
		errorToWeight: errorToWeight,
		errorToBias:   make([]float64, f.KernelCount),
		netToWeight:   newO2W(),
		netToInput:    newO2W(),
		outputToNet:   core.NewData(f.OutputSize),
	}
}

func (f *convolutionContext) Forward(prev core.Context) {
	f.input = prev.GetOutput()
	f.output.ForEachIndex(func(kernel, x, y int, value float64) {
		net := f.layer.Bias[kernel]
		for i := 0; i < f.layer.KernelSize; i++ {
			for j := 0; j < f.layer.KernelSize; j++ {
				for z := 0; z < f.layer.InputSize[0]; z++ {
					inputX := x + i - f.layer.Padding
					inputY := y + j - f.layer.Padding
					w := f.layer.Weight[kernel].Value[z][i][j]
					isPadding := inputX < 0 || inputX >= f.layer.InputSize[1] || inputY < 0 || inputY >= f.layer.InputSize[2]
					inputValue := 0.0
					if !isPadding {
						inputValue = f.input.Value[z][inputX][inputY]
					}
					net += inputValue * w
					if !isPadding {
						f.netToInput[kernel][x][y].Value[z][i][j] = w
						f.netToWeight[kernel][x][y].Value[z][i][j] = inputValue
					}
				}
			}
		}
		output, partial := f.layer.Activation.Active(net)
		f.output.Value[kernel][x][y] = output
		f.outputToNet.Value[kernel][x][y] = partial
	})
}

func (f *convolutionContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
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

func (f *convolutionContext) GetOutput() core.Data {
	return f.output
}

func (f *convolutionContext) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.InputSize)
	f.errorToOutput.ForEachIndex(func(kernel, x, y int, v float64) {
		for i := 0; i < f.layer.KernelSize; i++ {
			for j := 0; j < f.layer.KernelSize; j++ {
				for z := 0; z < f.layer.InputSize[0]; z++ {
					inputX := x + i - f.layer.Padding
					inputY := y + j - f.layer.Padding
					isPadding := inputX < 0 || inputX >= f.layer.InputSize[1] || inputY < 0 || inputY >= f.layer.InputSize[2]
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

func (f *Convolution) GetOutputSize() core.Size {
	return f.OutputSize
}