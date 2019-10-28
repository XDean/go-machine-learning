package layer

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

func init() {
	persistent.Register(new(Pooling))
}

type (
	PoolingType int
	Pooling     struct {
		Type    PoolingType
		Size    int // F
		Stride  int // S
		Padding int // P

		InputSize [3]int // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		OutputSize [3]int // D1 * W2 * H2
	}

	poolingContext struct {
		layer         *Pooling
		input         Data
		output        Data
		errorToOutput Data     // output
		errorToInput  Data     // input
		outputToInput [][]Data // F * F * output
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

func (f *Pooling) Init(prev, next Layer) {
	inputSize := prev.GetOutputSize()
	f.InputSize = inputSize
	f.OutputSize = [3]int{
		f.InputSize[0],
		(f.InputSize[1]+2*f.Padding-f.Size)/f.Stride + 1,
		(f.InputSize[2]+2*f.Padding-f.Size)/f.Stride + 1,
	}
}

func (f *Pooling) Learn(ctxs []Context) {
	// do nothing
}

func (f *Pooling) NewContext() Context {
	outputToInput := make([][]Data, f.Size)
	for i := range outputToInput {
		outputToInput[i] = make([]Data, f.Size)
		for j := range outputToInput[i] {
			outputToInput[i][j] = NewData(f.OutputSize)
		}
	}
	return &poolingContext{
		layer:         f,
		output:        NewData(f.OutputSize),
		errorToOutput: NewData(f.OutputSize),
		outputToInput: outputToInput,
	}
}

func (f *poolingContext) Forward(prev Context) {
	f.input = prev.GetOutput()
	f.output.MapIndex(func(dep, x, y int, _ float64) float64 {
		maxIndex := [2]int{0, 0}
		maxValue := -math.MaxFloat64
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
				switch f.layer.Type {
				case POOL_MAX:
					if inputValue > maxValue {
						f.outputToInput[maxIndex[0]][maxIndex[1]].Value[dep][x][y] = 0
						f.outputToInput[i][j].Value[dep][x][y] = 1
						maxValue = inputValue
						maxIndex[0] = i
						maxIndex[1] = j
					} else {
						f.outputToInput[i][j].Value[dep][x][y] = 0
					}
				case POOL_AVG:
					sum += inputValue
				case POOL_SUM:
					sum += inputValue
				}
			}
		}
		switch f.layer.Type {
		case POOL_MAX:
			return maxValue
		case POOL_AVG:
			return sum / float64(f.layer.Size*f.layer.Size)
		case POOL_SUM:
			return sum
		default:
			panic("Unknown pooling type")
		}
	})
}

func (f *poolingContext) Backward(next Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *poolingContext) GetOutput() Data {
	return f.output
}

func (f *poolingContext) GetErrorToInput() Data {
	result := NewData(f.layer.InputSize)
	avg := 1.0 / float64(f.layer.Size*f.layer.Size)
	f.errorToOutput.ForEachIndex(func(dep, x, y int, value float64) {
		for i := 0; i < f.layer.Size; i++ {
			for j := 0; j < f.layer.Size; j++ {
				inputX := x + i - f.layer.Padding
				inputY := y + j - f.layer.Padding
				isPadding := inputX < 0 || inputX >= f.layer.InputSize[0] || inputY < 0 || inputY >= f.layer.InputSize[1]
				if isPadding {
					continue
				}
				switch f.layer.Type {
				case POOL_MAX:
					result.Value[dep][inputX][inputY] += value * f.outputToInput[i][j].Value[dep][x][y]
				case POOL_AVG:
					result.Value[dep][inputX][inputY] += value * avg
				case POOL_SUM:
					result.Value[dep][inputX][inputY] += value
				}
			}
		}
	})
	return result
}

func (f *Pooling) GetOutputSize() Size {
	return f.OutputSize
}
