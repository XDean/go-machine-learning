package model

import (
	"encoding/gob"
)

type (
	Layer interface {
		Name() string

		Forward()  // call next
		Backward() // call prev
		Learn()    // call prev

		GetInput() Data         // i (prev output)
		GetOutput() Data        // o
		GetErrorToOutput() Data // ∂E / ∂a [output] a
		GetOutputToInput() Data // ∂a / ∂i [output, input] a * i

		GetInputSize() []uint // nil means no constraint
		GetOutputSize() []uint

		SetPrev(Layer)
		SetNext(Layer)
		GetPrev() Layer
		GetNext() Layer

		Save(writer *gob.Encoder) error
		Load(reader *gob.Decoder) error
	}

	BaseLayer struct {
		Prev, Next Layer
	}
)

func (bl *BaseLayer) SetPrev(l Layer) {
	bl.Prev = l
}

func (bl *BaseLayer) SetNext(l Layer) {
	bl.Next = l
}

func (bl *BaseLayer) GetPrev() Layer {
	return bl.Prev
}

func (bl *BaseLayer) GetNext() Layer {
	return bl.Next
}
