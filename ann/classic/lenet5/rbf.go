package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(new(RBF))
}

var (
	asciiTable = [10][12]int{
		{0x00, 0x1E, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x3C, 0x00, 0x00},
		{0x00, 0x0C, 0x1C, 0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x00, 0x00},
		{0x00, 0x1C, 0x36, 0x36, 0x06, 0x0C, 0x0C, 0x18, 0x30, 0x3E, 0x00, 0x00},
		{0x00, 0x1C, 0x36, 0x06, 0x06, 0x1C, 0x06, 0x06, 0x36, 0x1C, 0x00, 0x00},
		{0x00, 0x18, 0x18, 0x18, 0x36, 0x36, 0x36, 0x3F, 0x06, 0x06, 0x00, 0x00},
		{0x00, 0x3E, 0x30, 0x30, 0x30, 0x3C, 0x06, 0x06, 0x36, 0x1C, 0x00, 0x00},
		{0x00, 0x0C, 0x18, 0x18, 0x30, 0x3C, 0x36, 0x36, 0x36, 0x1C, 0x00, 0x00},
		{0x00, 0x3E, 0x06, 0x06, 0x0C, 0x0C, 0x0C, 0x18, 0x18, 0x18, 0x00, 0x00},
		{0x00, 0x1C, 0x36, 0x36, 0x36, 0x1C, 0x36, 0x36, 0x36, 0x1C, 0x00, 0x00},
		{0x00, 0x1C, 0x36, 0x36, 0x36, 0x1E, 0x0C, 0x0C, 0x18, 0x30, 0x00, 0x00},
	}
	rbfInputSize  = core.Size{1, 1, 7 * 12}
	rbfOutputSize = core.Size{1, 1, 10}
)

type (
	RBF struct{}

	rbfContext struct {
		layer         *RBF
		output        core.Data
		errorToOutput core.Data
		outputToInput [10][84]float64
	}
)

func NewRBF() *RBF {
	return &RBF{}
}

func (f *RBF) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	if inputSize != rbfInputSize {
		panic("Input must be 84")
	}
}

func (f *RBF) Learn(ctxs []core.Context) {
	// do nothing
}

func (f *RBF) NewContext() core.Context {
	return &rbfContext{
		layer:         f,
		output:        core.NewData(rbfOutputSize),
		outputToInput: [10][84]float64{},
	}
}

func (f *rbfContext) Forward(prev core.Context) {
	input := prev.GetOutput()
	f.output.MapIndex(func(_, _, o int, _ float64) float64 {
		sum := 0.0
		input.ForEachIndex(func(_, _, i int, value float64) {
			asciiStub := asciiTable[o][i/7]
			asciiValue := (asciiStub >> uint(i%7)) & 1
			delta := value - float64(asciiValue)
			if asciiValue == 0 {
				delta += 1
			}
			sum += delta * delta
			f.outputToInput[o][i] = 2 * delta
		})
		return sum
	})
}

func (f *rbfContext) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
}

func (f *rbfContext) GetOutput() core.Data {
	return f.output
}

func (f *rbfContext) GetErrorToInput() core.Data {
	result := core.NewData(rbfInputSize)
	result.MapIndex(func(_, _, i int, _ float64) float64 {
		sum := 0.0
		f.errorToOutput.ForEachIndex(func(_, _, o int, value float64) {
			sum += value * f.outputToInput[o][i]
		})
		return sum
	})
	return result
}

func (f *RBF) GetOutputSize() core.Size {
	return rbfOutputSize
}

func (f *RBF) Desc() core.Desc {
	return core.SimpleDesc{Name: "Output RBF"}
}
