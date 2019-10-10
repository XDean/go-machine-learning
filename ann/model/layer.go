package model

type (
	Layer interface {
		Forward(input *Data) (output *Data)
		Backward(nextError *Data) (myError *Data)
		Learn(ratio float64)

		GetInput() *Data
		GetOutput() *Data
		GetError() *Data

		SetPrev(*Layer)
		SetNext(*Layer)
		GetPrev() *Layer
		GetNext() *Layer
	}

	BaseLayer struct {
		Prev, Next *Layer
	}
)

func (bl *BaseLayer) SetPrev(l *Layer) {
	bl.Prev = l
}

func (bl *BaseLayer) SetNext(l *Layer) {
	bl.Next = l
}

func (bl *BaseLayer) GetPrev() *Layer {
	return bl.Prev
}

func (bl *BaseLayer) GetNext() *Layer {
	return bl.Next
}
