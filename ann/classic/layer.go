package classic

import . "github.com/XDean/go-machine-learning/ann/model"

type FullLayer struct {
	BaseLayer

	Size   uint
	Weight [][]float64
}

func NewFullLayer(size uint) *FullLayer {
	return &FullLayer{
		Size: size,
	}
}

func (f *FullLayer) Forward(input Data) {
	f.Next.Forward(f.GetOutput())
}

func (f *FullLayer) Backward(Error float64, Target Data) {
	f.Prev.Backward(Error, Target)
}

func (f *FullLayer) Learn() {
	panic("implement me")
}

func (f *FullLayer) GetInput() Data {
	panic("implement me")
}

func (f *FullLayer) GetOutput() Data {
	panic("implement me")
}

func (f *FullLayer) GetError() Data {
	panic("implement me")
}

func (f *FullLayer) GetInputSize() []uint {
	return nil
}

func (f *FullLayer) GetOutputSize() []uint {
	return []uint{f.Size}
}

func (f *FullLayer) SetPrev(l Layer) {
	f.BaseLayer.SetPrev(l)
}
