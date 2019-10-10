package model

type (
	Layer interface {
		Forward()                            // call next
		Backward(error float64, target Data) // call prev
		Learn()                              // call prev

		GetInput() Data
		GetOutput() Data
		GetWeight() Data
		GetError() Data

		GetInputSize() []uint // nil means no constraint
		GetOutputSize() []uint

		SetPrev(Layer)
		SetNext(Layer)
		GetPrev() Layer
		GetNext() Layer
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