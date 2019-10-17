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

	errorToInput := data.NewData(size...)
	errorToInput.MapIndex(func(inputIndex []int, value float64) float64 {
		sum := 0.0
		errorToOutput.ForEachIndex(func(outputIndex []int, value float64) {
			sum += value * outputToInput.GetValue(append(outputIndex, inputIndex...)...)
		})
		return sum
	})
	return errorToInput
}
