package layer

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	"github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(new(DimAdd))
}

type DimAdd struct {
	model.BaseLayer

	Dim int

	InputSize  []int
	OutputSize []int

	input        data.Data
	output       data.Data
	errorToInput data.Data
}

func NewDimAdd(dim int) *DimAdd {
	return &DimAdd{Dim: dim}
}

func (d *DimAdd) Init() {
	if d.InputSize == nil {
		d.InputSize = d.GetPrev().GetOutputSize()
		addDim := make([]int, d.Dim)
		for i := range addDim {
			addDim[i] = 1
		}
		d.OutputSize = append(d.InputSize, addDim...)
	}
	d.errorToInput = data.NewData(d.GetPrev().GetOutputSize())
}

func (d *DimAdd) Forward() {
	d.input = d.GetPrev().GetOutput()
	d.output = data.NewDimAdd(d.input, d.Dim)
}

func (d *DimAdd) Backward() {
	d.GetNext().GetErrorToInput().ForEachIndex(func(indexes []int, value float64) {
		d.errorToInput.SetValue(value, indexes[:len(indexes)-d.Dim])
	})
}

func (d *DimAdd) Learn() {
	// do nothing
}

func (d *DimAdd) GetInput() data.Data {
	return d.input
}

func (d *DimAdd) GetOutput() data.Data {
	return d.output
}

func (d *DimAdd) GetErrorToInput() data.Data {
	return d.GetNext().GetErrorToInput()
}

func (d *DimAdd) GetOutputSize() []int {
	return d.OutputSize
}
