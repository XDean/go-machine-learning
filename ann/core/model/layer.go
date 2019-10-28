package model

type (
	Layer interface {
		Init(prev, next Layer)
		Learn([]Context)
		NewContext() Context
		GetOutputSize() Size
	}

	Context interface {
		Forward(prev Context)
		Backward(next Context)

		GetOutput() Data       // o
		GetErrorToInput() Data // ∂E / ∂a
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
