package layer

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
	"github.com/XDean/go-machine-learning/ann/core/model"
)

type DimAdd struct {
	model.BaseLayer
}

func (d DimAdd) Init() {
	// do nothing
}

func (d DimAdd) Forward() {
	// do nothing
}

func (d DimAdd) Backward() {
	// do nothing
}

func (d DimAdd) Learn() {
	// do nothing
}

func (d DimAdd) GetInput() data.Data {
	return d.GetNext().GetInput()
}

func (d DimAdd) GetOutput() data.Data {
	return d.GetPrev().GetOutput()
}

func (d DimAdd) GetErrorToInput() data.Data {
	return d.GetNext().GetErrorToInput()
}

func (d DimAdd) GetOutputSize() []int {
	return d.GetPrev().GetOutputSize()
}
