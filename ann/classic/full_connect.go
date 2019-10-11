package classic

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
	. "github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(func() persistent.Persistent {
		return NewFullLayer(FullLayerConfig{})
	})
}

type (
	FullLayer struct {
		BaseLayer

		Weight base.Data // a * i
		Bias   float64   // TODO, no learning now

		Input          base.Data // i * 1
		Output         base.Data // a * 1
		ErrorToOutput  base.Data // a * 1, ∂E / ∂a
		OutputToInput  base.Data // a * i, ∂a / ∂i
		OutputToWeight base.Data // a * i, ∂a / ∂w

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
		Activation:    Sigmoid{},
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

func (f *FullLayer) Init() {
	f.Weight = f.WeightInit.Init(base.NewData(append([]uint{f.Size}, f.Prev.GetOutputSize()...)...))
}

func (f *FullLayer) Forward() {
	f.Input = f.Prev.GetOutput()
	f.Output = base.NewData(f.Size)
	f.ErrorToOutput = base.NewData(f.Size)
	f.OutputToInput = base.NewData(append([]uint{f.Size}, f.Input.GetSize()...)...)
	f.OutputToWeight = base.NewData(append([]uint{f.Size}, f.Input.GetSize()...)...)

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
}

func (f *FullLayer) Backward() {
	f.ErrorToOutput = ErrorToInput(f.Next) // next layer's input is this layer's output
}

func (f *FullLayer) Learn() {
	f.Weight.ForEach(func(index []uint, value float64) {
		f.Weight.SetValue(value-f.LearningRatio*f.ErrorToOutput.GetValue(index[0])*f.OutputToWeight.GetValue(index...), index...)
	})
}

func (f *FullLayer) GetInput() base.Data {
	return f.Input
}

func (f *FullLayer) GetOutput() base.Data {
	return f.Output
}

func (f *FullLayer) GetErrorToOutput() base.Data {
	return f.ErrorToOutput
}

func (f *FullLayer) GetOutputToInput() base.Data {
	return f.OutputToInput
}

func (f *FullLayer) GetInputSize() []uint {
	return nil
}

func (f *FullLayer) GetOutputSize() []uint {
	return []uint{f.Size}
}

func (f *FullLayer) Save(writer *gob.Encoder) (err error) {
	defer base.RecoverNoError(&err)
	base.NoError(writer.Encode(f.Size))
	base.NoError(writer.Encode(f.Weight))
	base.NoError(writer.Encode(f.Bias))
	base.NoError(writer.Encode(f.LearningRatio))
	base.NoError(persistent.Save(writer, f.Activation))
	base.NoError(persistent.Save(writer, f.WeightInit))
	return nil
}

func (f *FullLayer) Load(reader *gob.Decoder) (err error) {
	defer base.RecoverNoError(&err)
	base.NoError(reader.Decode(&f.Size))
	base.NoError(reader.Decode(&f.Weight))
	base.NoError(reader.Decode(&f.Bias))
	base.NoError(reader.Decode(&f.LearningRatio))

	bean, err := persistent.Load(reader)
	base.NoError(err)
	if actual, ok := bean.(Activation); ok {
		f.Activation = actual
	} else {
		return persistent.TypeError("Activation", bean)
	}

	bean, err = persistent.Load(reader)
	base.NoError(err)
	if actual, ok := bean.(WeightInit); ok {
		f.WeightInit = actual
	} else {
		return persistent.TypeError("WeightInit", bean)
	}
	return nil
}
