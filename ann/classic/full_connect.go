package classic

import (
	"encoding/gob"
	. "github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/model/persistent"
)

func init() {
	persistent.Register(func() persistent.Persistent {
		return NewFullLayer(FullLayerConfig{})
	})
}

type (
	FullLayer struct {
		BaseLayer

		Weight Data // a * i
		Bias   float64

		Input          Data // i * 1
		Output         Data // a * 1
		ErrorToOutput  Data // a * 1, ∂E / ∂a
		OutputToInput  Data // a * i, ∂a / ∂i
		OutputToWeight Data // a * i, ∂a / ∂w

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

var (
	DefaultConfig = FullLayerConfig{
		Size:          10,
		Activation:    ReLU{},
		LearningRatio: 0.1,
		WeightInit:    RandomInit{},
	}
)

func NewFullLayer(config FullLayerConfig) *FullLayer {
	if config.Size == 0 {
		config.Size = DefaultConfig.Size
	}
	if config.Activation == nil {
		config.Activation = DefaultConfig.Activation
	}
	if config.LearningRatio == 0 {
		config.LearningRatio = DefaultConfig.LearningRatio
	}
	if config.WeightInit == nil {
		config.WeightInit = DefaultConfig.WeightInit
	}
	return &FullLayer{
		Size:          config.Size,
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *FullLayer) Name() string {
	return "Full Connect Layer"
}

func (f *FullLayer) Forward() {
	input := f.Prev.GetOutput()

	f.Output = NewData(f.Size)
	f.ErrorToOutput = NewData(f.Size)
	f.OutputToInput = NewData(append([]uint{f.Size}, input.GetSize()...)...)

	f.Output.ForEach(func(outputIndex []uint, _ float64) {
		net := 0.0
		f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []uint, w float64) {
			net += w * f.Input.GetValue(inputIndex...)
		})
		output, partial := f.Activation.Active(net)
		f.Output.SetValue(output, outputIndex...)
		f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []uint, w float64) {
			f.OutputToInput.SetValue(partial*w, append(outputIndex, inputIndex...)...)
			f.OutputToWeight.SetValue(partial*f.Input.GetValue(inputIndex...), append(outputIndex, inputIndex...)...)
		})
	})

	f.Next.Forward()
}

func (f *FullLayer) Backward() {
	f.ErrorToOutput = ErrorToInput(f.Next) // next layer's input is this layer's output
	f.Prev.Backward()
}

func (f *FullLayer) Learn() {
	f.Weight.ForEach(func(index []uint, value float64) {
		f.Weight.SetValue(value-f.LearningRatio*f.ErrorToOutput.GetValue(index[0])*f.OutputToWeight.GetValue(index...), index...)
	})
}

func (f *FullLayer) GetInput() Data {
	return f.Input
}

func (f *FullLayer) GetOutput() Data {
	return f.Output
}

func (f *FullLayer) GetErrorToOutput() Data {
	return f.ErrorToOutput
}

func (f *FullLayer) GetOutputToInput() Data {
	return f.OutputToInput
}

func (f *FullLayer) GetInputSize() []uint {
	return nil
}

func (f *FullLayer) GetOutputSize() []uint {
	return []uint{f.Size}
}

func (f *FullLayer) SetPrev(l Layer) {
	f.BaseLayer.SetPrev(l)
	f.Weight = f.WeightInit.Init(NewData(append([]uint{f.Size}, l.GetOutputSize()...)...))
}

func (f *FullLayer) Save(writer *gob.Encoder) (err error) {
	defer RecoverNoError(&err)
	NoError(writer.Encode(f.Size))
	NoError(writer.Encode(f.Weight))
	NoError(writer.Encode(f.Bias))
	NoError(writer.Encode(f.LearningRatio))
	NoError(persistent.Save(writer, f.Activation))
	NoError(persistent.Save(writer, f.WeightInit))
	return nil
}

func (f *FullLayer) Load(reader *gob.Decoder) (err error) {
	defer RecoverNoError(&err)
	NoError(reader.Decode(&f.Size))
	NoError(reader.Decode(&f.Weight))
	NoError(reader.Decode(&f.Bias))
	NoError(reader.Decode(&f.LearningRatio))

	bean, err := persistent.Load(reader)
	NoError(err)
	if actual, ok := bean.(Activation); ok {
		f.Activation = actual
	} else {
		return persistent.TypeError("Activation", bean)
	}

	bean, err = persistent.Load(reader)
	NoError(err)
	if actual, ok := bean.(WeightInit); ok {
		f.WeightInit = actual
	} else {
		return persistent.TypeError("WeightInit", bean)
	}
	return nil
}