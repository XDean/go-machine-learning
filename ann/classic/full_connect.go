package classic

import (
	"github.com/XDean/go-machine-learning/ann/base"
	. "github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"sync"
)

func init() {
	persistent.Register(NewFullLayer(FullLayerDefaultConfig))
}

type (
	FullLayer struct {
		BaseLayer

		Weight base.Data // a * i
		Bias   float64   // TODO, no learning now

		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit

		input          base.Data // i * 1
		output         base.Data // a * 1
		errorToOutput  base.Data // a * 1, ∂E / ∂a
		outputToInput  base.Data // a * i, ∂a / ∂i
		outputToWeight base.Data // a * i, ∂a / ∂w
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

func (f *FullLayer) Init() {
	f.Weight = f.WeightInit.Init(base.NewData(append([]int{f.Size}, f.GetPrev().GetOutputSize()...)...))
}

func (f *FullLayer) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output = base.NewData(f.Size)
	f.errorToOutput = base.NewData(f.Size)
	f.outputToInput = base.NewData(append([]int{f.Size}, f.input.GetSize()...)...)
	f.outputToWeight = base.NewData(append([]int{f.Size}, f.input.GetSize()...)...)

	wg := sync.WaitGroup{}
	f.output.ForEach(func(outputIndex []int, _ float64) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			net := 0.0
			f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []int, w float64) {
				net += w * f.input.GetValue(inputIndex...)
			})
			output, partial := f.Activation.Active(net)
			f.output.SetValue(output, outputIndex...)
			f.Weight.GetData(outputIndex...).ForEach(func(inputIndex []int, w float64) {
				f.outputToInput.SetValue(partial*w, append(outputIndex, inputIndex...)...)
				f.outputToWeight.SetValue(partial*f.input.GetValue(inputIndex...), append(outputIndex, inputIndex...)...)
			})
		}()
	})
	wg.Wait()
}

func (f *FullLayer) Backward() {
	f.errorToOutput = ErrorToInput(f.GetNext()) // next layer's input is this layer's output
}

func (f *FullLayer) Learn() {
	//fmt.Println("weight", f.Weight.ToArray()[:10])
	//fmt.Println("ErrorToOutput", f.ErrorToOutput.ToArray()[:10])
	f.Weight.ForEach(func(index []int, value float64) {
		f.Weight.SetValue(value-f.LearningRatio*f.errorToOutput.GetValue(index[0])*f.outputToWeight.GetValue(index...), index...)
	})
}

func (f *FullLayer) GetInput() base.Data {
	return f.input
}

func (f *FullLayer) GetOutput() base.Data {
	return f.output
}

func (f *FullLayer) GetErrorToOutput() base.Data {
	return f.errorToOutput
}

func (f *FullLayer) GetOutputToInput() base.Data {
	return f.outputToInput
}

func (f *FullLayer) GetInputSize() []int {
	return nil
}

func (f *FullLayer) GetOutputSize() []int {
	return []int{f.Size}
}
