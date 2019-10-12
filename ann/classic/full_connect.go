package classic

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
	. "github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/XDean/go-machine-learning/ann/util"
	"sync"
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

		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}

	FullLayerConfig struct {
		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}
)

var (
	FullLayerDefaultConfig = FullLayerConfig{
		Size:          10,
		Activation:    Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &RandomInit{Range: 1},
	}
)

func NewFullLayer(config FullLayerConfig) *FullLayer {
	if config.Size == 0 {
		config.Size = FullLayerDefaultConfig.Size
	}
	if config.Activation == nil {
		config.Activation = FullLayerDefaultConfig.Activation
	}
	if config.LearningRatio == 0 {
		config.LearningRatio = FullLayerDefaultConfig.LearningRatio
	}
	if config.WeightInit == nil {
		config.WeightInit = FullLayerDefaultConfig.WeightInit
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
	f.Weight = f.WeightInit.Init(base.NewData(append([]int{f.Size}, f.Prev.GetOutputSize()...)...))
}

func (f *FullLayer) Forward() {
	f.Input = f.Prev.GetOutput()
	f.Output = base.NewData(f.Size)
	f.ErrorToOutput = base.NewData(f.Size)
	f.OutputToInput = base.NewData(append([]int{f.Size}, f.Input.GetSize()...)...)
	f.OutputToWeight = base.NewData(append([]int{f.Size}, f.Input.GetSize()...)...)

	wg := sync.WaitGroup{}
	f.Output.ForEach(func(outputIndex []int, _ float64) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			net := 0.0
			f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []int, w float64) {
				net += w * f.Input.GetValue(inputIndex...)
			})
			output, partial := f.Activation.Active(net)
			f.Output.SetValue(output, outputIndex...)
			f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []int, w float64) {
				f.OutputToInput.SetValue(partial*w, append(outputIndex, inputIndex...)...)
				f.OutputToWeight.SetValue(partial*f.Input.GetValue(inputIndex...), append(outputIndex, inputIndex...)...)
			})
		}()
	})
	wg.Wait()
}

func (f *FullLayer) Backward() {
	f.ErrorToOutput = ErrorToInput(f.Next) // next layer's input is this layer's output
}

func (f *FullLayer) Learn() {
	//fmt.Println("weight", f.Weight.ToArray()[:10])
	//fmt.Println("ErrorToOutput", f.ErrorToOutput.ToArray()[:10])
	f.Weight.ForEach(func(index []int, value float64) {
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

func (f *FullLayer) GetInputSize() []int {
	return nil
}

func (f *FullLayer) GetOutputSize() []int {
	return []int{f.Size}
}

func (f *FullLayer) Save(writer *gob.Encoder) (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(writer.Encode(f.Size))
	util.NoError(writer.Encode(f.Weight))
	util.NoError(writer.Encode(f.Bias))
	util.NoError(writer.Encode(f.LearningRatio))
	util.NoError(persistent.Save(writer, f.Activation))
	util.NoError(persistent.Save(writer, f.WeightInit))
	return nil
}

func (f *FullLayer) Load(reader *gob.Decoder) (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(reader.Decode(&f.Size))
	util.NoError(reader.Decode(&f.Weight))
	util.NoError(reader.Decode(&f.Bias))
	util.NoError(reader.Decode(&f.LearningRatio))

	bean, err := persistent.Load(reader)
	util.NoError(err)
	if actual, ok := bean.(Activation); ok {
		f.Activation = actual
	} else {
		return persistent.TypeError("Activation", bean)
	}

	bean, err = persistent.Load(reader)
	util.NoError(err)
	if actual, ok := bean.(WeightInit); ok {
		f.WeightInit = actual
	} else {
		return persistent.TypeError("WeightInit", bean)
	}
	return nil
}
