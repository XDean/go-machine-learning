package model

import (
	"github.com/XDean/go-machine-learning/ann/base"
)

type (
	Layer interface {
		Init() // after set prev and next

		Forward()  // call next
		Backward() // call prev
		Learn()    // call prev

		GetInput() base.Data         // i (prev output)
		GetOutput() base.Data        // o
		GetErrorToOutput() base.Data // ∂E / ∂a [output] a
		GetOutputToInput() base.Data // ∂a / ∂i [output, input] a * i

		GetInputSize() []int // nil means no constraint
		GetOutputSize() []int

		SetPrev(Layer)
		SetNext(Layer)
		GetPrev() Layer
		GetNext() Layer
	}

	BaseLayer struct {
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

func ErrorToInput(l Layer) base.Data {
	errorToOutput := l.GetErrorToOutput()
	outputToInput := l.GetOutputToInput()

	os := errorToOutput.GetSize()
	ois := outputToInput.GetSize()
	size := ois[len(os):]

	errorToInput := base.NewData(size...)
	errorToInput.ForEach(func(inputIndex []int, value float64) {
		sum := 0.0
		errorToOutput.ForEach(func(outputIndex []int, value float64) {
			sum += value * outputToInput.GetValue(append(outputIndex, inputIndex...)...)
		})
		errorToInput.SetValue(sum, inputIndex...)
	})
	return errorToInput
}
