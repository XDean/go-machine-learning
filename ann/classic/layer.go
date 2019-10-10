package classic

import . "github.com/XDean/go-machine-learning/ann/model"

type FullLayer struct {
	BaseLayer
	Size uint
}

func (f *FullLayer) Forward(input *Data) (output *Data) {
	panic("implement me")
}

func (f *FullLayer) Backward(nextError *Data) (myError *Data) {
	panic("implement me")
}

func (f *FullLayer) Learn(ratio float64) {
	panic("implement me")
}

func (f *FullLayer) GetInput() *Data {
	panic("implement me")
}

func (f *FullLayer) GetOutput() *Data {
	panic("implement me")
}

func (f *FullLayer) GetError() *Data {
	panic("implement me")
}
