package layer

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"github.com/XDean/go-machine-learning/ann/core/util"
)

func init() {
	persistent.Register(new(Pooling))
}

type (
	PoolingType int
	Pooling     struct {
		BaseLayer

		Type    PoolingType
		Size    int
		Padding int
		Stride  int

		InputSize  [3]int
		OutputSize [3]int
		noDepth    bool

		input         data.Data
		output        data.Data
		errorToOutput data.Data // output
		errorToInput  data.Data // input
		outputToInput data.Data // output * input
	}

	PoolingConfig struct {
		Type    PoolingType
		Size    int
		Padding int
		Stride  int
	}
)

const (
	POOL_MAX PoolingType = iota + 1
	POOL_AVG
	POOL_SUM
)

var (
	PoolingDefaultConfig = PoolingConfig{
		Type:    POOL_MAX,
		Size:    2,
		Padding: 0,
		Stride:  1,
	}
)

func NewPooling(config PoolingConfig) *Pooling {
	if config.Type == 0 {
		config.Type = PoolingDefaultConfig.Type
	}
	if config.Size == 0 {
		config.Size = PoolingDefaultConfig.Size
	}
	if config.Stride == 0 {
		config.Stride = PoolingDefaultConfig.Stride
	}
	return &Pooling{
		Type:    config.Type,
		Size:    config.Size,
		Stride:  config.Stride,
		Padding: config.Padding,
	}
}

func (f *Pooling) Init() {
	inputSize := f.GetPrev().GetOutputSize()
	util.MustTrue(len(inputSize) == 2 || len(inputSize) == 3)
	switch len(inputSize) {
	case 2:
		f.noDepth = true
		f.InputSize = [3]int{inputSize[0], inputSize[1], 1}
	case 3:
		f.noDepth = false
		f.InputSize = [3]int{inputSize[0], inputSize[1], inputSize[2]}
	default:
		util.MustTrue(false, "Input must be 2 or 3 dim")
	}
	f.OutputSize = [3]int{
		(f.InputSize[0]+2*f.Padding-f.Size)/f.Stride + 1,
		(f.InputSize[1]+2*f.Padding-f.Size)/f.Stride + 1,
		f.InputSize[2],
	}
}

func (f *Pooling) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output = data.NewData(f.OutputSize[:]...)
	f.errorToOutput = data.NewData(f.OutputSize[:]...)
	f.outputToInput = data.NewData(append(f.OutputSize[:], func() []int {
		if f.noDepth {
			return f.InputSize[:2]
		} else {
			return f.InputSize[:]
		}
	}()...)...)

	f.output.MapIndex(func(outputIndex []int, _ float64) float64 {
		depth := outputIndex[2]
		maxIndex := []int{0, 0}
		maxValue := 0.0
		sum := 0.0
		for i := 0; i < f.Size; i++ {
			for j := 0; j < f.Size; j++ {
				inputX := outputIndex[0] + i - f.Padding
				inputY := outputIndex[1] + j - f.Padding
				inputValue := func() (result float64) {
					if inputX < 0 || inputX >= f.InputSize[0] || inputY < 0 || inputY >= f.InputSize[1] {
						return 0.0
					} else if f.input.GetDim() == 2 {
						return f.input.GetValue(inputX, inputY)
					} else {
						return f.input.GetValue(inputX, inputY, depth)
					}
				}()
				switch f.Type {
				case POOL_MAX:
					if inputValue > maxValue {
						maxValue = inputValue
						maxIndex[0] = i
						maxIndex[1] = j
					}
				case POOL_AVG | POOL_SUM:
					sum += inputValue
				}
			}
		}
		switch f.Type {
		case POOL_MAX:
			if f.noDepth {
				f.outputToInput.SetValue(1, append(outputIndex, maxIndex...)...)
			} else {
				f.outputToInput.SetValue(1, append(outputIndex, maxIndex[0], maxIndex[1], depth)...)
			}
			return maxValue
		case POOL_AVG:
			data.MapSub(f.outputToInput, func(indexes []int, value float64) float64 {
				return 1 / float64(f.Size*f.Size)
			}, outputIndex...)
			return sum / float64(f.Size*f.Size)
		case POOL_SUM:
			data.MapSub(f.outputToInput, func(indexes []int, value float64) float64 {
				return 1
			}, outputIndex...)
			return sum
		}
		return 0
	})
}

func (f *Pooling) Backward() {
	f.errorToOutput = f.GetNext().GetErrorToInput()
	f.errorToInput = ErrorToInput(f.errorToOutput, f.outputToInput)
}

func (f *Pooling) Learn() {
	// do nothing
}

func (f *Pooling) GetInput() data.Data {
	return f.input
}

func (f *Pooling) GetOutput() data.Data {
	return f.output
}

func (f *Pooling) GetErrorToInput() data.Data {
	return f.errorToInput
}

func (f *Pooling) GetOutputSize() []int {
	return f.OutputSize[:]
}
