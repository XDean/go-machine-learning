package layer

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math"
)

func init() {
	persistent.Register(new(SoftMax))
}

type (
	SoftMax struct {
		Size core.Size
	}
	softMaxContext struct {
		layer         *SoftMax
		output        core.Data
		errorToOutput core.Data
	}
)

func NewSoftMax() *SoftMax {
	return &SoftMax{}
}

func (f *SoftMax) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	f.Size = inputSize
}

func (f *SoftMax) Learn(ctxs []core.Context) {
	// do nothing
}

func (f *SoftMax) NewContext() core.Context {
	return &softMaxContext{
		layer:  f,
		output: core.NewData(f.Size),
	}
}

func (f *softMaxContext) Forward(prev core.Context) {
	input := prev.GetOutput()
	max := 0.0
	input.ForEach(func(value float64) {
		if value > max {
			max = value
		}
	})
	sum := 0.0
	input.ForEachIndex(func(i, j, k int, value float64) {
		exp := math.Exp(value - max)
		sum += exp
		f.output.Value[i][j][k] = exp
	})
	f.output.Map(func(value float64) float64 {
		return value / sum
	})
}

func (f *softMaxContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *softMaxContext) GetOutput() core.Data {
	return f.output
}

func (f *softMaxContext) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.Size)
	result.MapIndex(func(i1, j1, k1 int, _ float64) float64 {
		sum := 0.0
		f.errorToOutput.ForEachIndex(func(i2, j2, k2 int, value float64) {
			outputValue := f.output.Value[i2][j2][k2]
			if i1 == i2 && j1 == j2 && k1 == k2 {
				sum += value * outputValue * (1 - outputValue)
			} else {
				sum += value * -outputValue * f.output.Value[i1][j1][k1]
			}
		})
		return sum
	})
	return result
}

func (f *SoftMax) GetOutputSize() core.Size {
	return f.Size
}
