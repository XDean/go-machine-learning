package classic

import (
	. "github.com/XDean/go-machine-learning/ann/model"
)

type (
	FullLayer struct {
		BaseLayer

		Weight Data // i * j
		Bias   float64

		Input   Data // i * 1
		Output  Data // j * 1
		Partial Data // j * 1, D_out / D_net
		Error   Data // i * j, D_error / D_w

		Size          uint
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}

	FullLayerConfig struct {
		Size          uint
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}
)

func NewFullLayer(config FullLayerConfig) *FullLayer {
	if config.Size == 0 {
		panic("Size must be specified")
	}
	if config.Activation == nil {
		config.Activation = ReLU
	}
	if config.LearningRatio == 0 {
		config.LearningRatio = 0.1
	}
	if config.WeightInit == nil {
		config.WeightInit = RandomInit()
	}
	return &FullLayer{
		Size:          config.Size,
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *FullLayer) Forward() {
	f.Input = f.Prev.GetOutput().ToDim(1)
	f.Output = NewData(f.Size).Fill(f.Bias)
	f.Partial = NewData(f.Size)

	f.Input.ForEach(func(i []uint, prev float64) {
		f.Weight.GetData(i[0]).ForEach(func(j []uint, w float64) {
			f.Output.SetValue(f.Output.GetValue(j[0])+w*prev, j[0])
		})
	})

	f.Output.ForEach(func(index []uint, value float64) {
		output, partial := f.Activation.Active(value)
		f.Output.SetValue(output, index...)
		f.Partial.SetValue(partial, index...)
	})

	f.Next.Forward()
}

func (f *FullLayer) Backward(error float64, target Data) {
	if f.Next == nil {
		f.Error.ForEach(func(index []uint, value float64) {
			//i, j := index[0], index[1]

		})
	} else {

	}

	f.Prev.Backward(error, target)
}

func (f *FullLayer) Learn() {
	panic("implement me")
}

func (f *FullLayer) GetInput() Data {
	return f.Input
}

func (f *FullLayer) GetOutput() Data {
	return f.Output
}

func (f *FullLayer) GetWeight() Data {
	return f.Weight
}

func (f *FullLayer) GetError() Data {
	return f.Error
}

func (f *FullLayer) GetInputSize() []uint {
	return nil
}

func (f *FullLayer) GetOutputSize() []uint {
	return []uint{f.Size}
}

func (f *FullLayer) SetPrev(l Layer) {
	f.BaseLayer.SetPrev(l)
	lastCount := SizeToCount(l.GetOutputSize()...)
	f.Weight = f.WeightInit(NewData(lastCount, f.Size))
	f.Error = NewData(lastCount, f.Size)
}
