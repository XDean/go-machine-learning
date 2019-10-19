package model

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
)

type (
	Layer interface {
		Init() // after set prev and next

		Forward()  // call next
		Backward() // call prev
		Learn()    // call prev

		GetInput() data.Data        // i (prev output)
		GetOutput() data.Data       // o
		GetErrorToInput() data.Data // ∂E / ∂a [output] a

		GetOutputSize() []int

		SetPrev(Layer)
		SetNext(Layer)
		GetPrev() Layer
		GetNext() Layer
	}

	BaseLayer struct {
		Ignore     bool // let gob work
		prev, next Layer
	}
)

func (bl *BaseLayer) SetPrev(l Layer) {
	bl.prev = l
}

func (bl *BaseLayer) SetNext(l Layer) {
	bl.next = l
}

func (bl *BaseLayer) GetPrev() Layer {
	return bl.prev
}

func (bl *BaseLayer) GetNext() Layer {
	return bl.next
}

func ErrorToInput(errorToOutput, outputToInput data.Data) data.Data {
	os := errorToOutput.GetSize()
	ois := outputToInput.GetSize()
	size := ois[len(os):]

	errorToInput := data.NewData(size)
	index := make([]int, len(ois))

	errorToInput.MapIndex(func(inputIndex []int, value float64) float64 {
		sum := 0.0
		copy(index[len(os):], inputIndex)
		errorToOutput.ForEachIndex(func(outputIndex []int, value float64) {
			copy(index, outputIndex)
			sum += value * outputToInput.GetValue(index)
		})
		return sum
	})
	return errorToInput
}

func ErrorToInputByFunc(errorToOutput data.Data, inputSize []int, f func(outputIndex, inputIndex []int) float64) data.Data {
	errorToInput := data.NewData(inputSize)
	errorToInput.MapIndex(func(inputIndex []int, value float64) float64 {
		sum := 0.0
		errorToOutput.ForEachIndex(func(outputIndex []int, value float64) {
			sum += value * f(outputIndex, inputIndex)
		})
		return sum
	})
	return errorToInput
}
