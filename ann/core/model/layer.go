package model

type (
	Layer interface {
		Init() // after set prev and next

		Forward()  // call next
		Backward() // call prev
		Learn()    // call prev

		GetOutput() Data       // o
		GetErrorToInput() Data // ∂E / ∂a

		GetOutputSize() Size

		SetPrev(Layer)
		SetNext(Layer)
		GetPrev() Layer
		GetNext() Layer
	}

	BaseLayer struct {
		Init       bool
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
