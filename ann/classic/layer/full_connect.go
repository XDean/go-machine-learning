package layer

import (
	. "github.com/XDean/go-machine-learning/ann/classic"
	"github.com/XDean/go-machine-learning/ann/core/data"
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"sync"
)

func init() {
	persistent.Register(new(FullConnect))
}

type (
	FullConnect struct {
		BaseLayer

		Weight data.Data // a * i
		Bias   float64   // TODO, no learning now

		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit

		input          data.Data // i * 1
		output         data.Data // a * 1
		errorToOutput  data.Data // a * 1, ∂E / ∂a
		outputToInput  data.Data // a * i, ∂a / ∂i
		outputToWeight data.Data // a * i, ∂a / ∂w
	}

	FullConnectConfig struct {
		Size          int
		Activation    Activation
		LearningRatio float64
		WeightInit    WeightInit
	}
)

var (
	FullLayerDefaultConfig = FullConnectConfig{
		Size:          10,
		Activation:    Sigmoid{},
		LearningRatio: 0.1,
		WeightInit:    &RandomInit{Range: 1},
	}
)

func NewFullConnect(config FullConnectConfig) *FullConnect {
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
	return &FullConnect{
		Size:          config.Size,
		Activation:    config.Activation,
		LearningRatio: config.LearningRatio,
		WeightInit:    config.WeightInit,
	}
}

func (f *FullConnect) Init() {
	if f.Weight == nil {
		f.Weight = f.WeightInit.Init(data.NewData(append([]int{f.Size}, f.GetPrev().GetOutputSize()...)...))
	}
}

func (f *FullConnect) Forward() {
	f.input = f.GetPrev().GetOutput()
	f.output = data.NewData(f.Size)
	f.errorToOutput = data.NewData(f.Size)
	f.outputToInput = data.NewData(append([]int{f.Size}, f.input.GetSize()...)...)
	f.outputToWeight = data.NewData(append([]int{f.Size}, f.input.GetSize()...)...)

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

func (f *FullConnect) Backward() {
	f.errorToOutput = ErrorToInput(f.GetNext()) // next layer's input is this layer's output
}

func (f *FullConnect) Learn() {
	//fmt.Println("weight", f.Weight.ToArray()[:10])
	//fmt.Println("ErrorToOutput", f.ErrorToOutput.ToArray()[:10])
	f.Weight.ForEach(func(index []int, value float64) {
		f.Weight.SetValue(value-f.LearningRatio*f.errorToOutput.GetValue(index[0])*f.outputToWeight.GetValue(index...), index...)
	})
}

func (f *FullConnect) GetInput() data.Data {
	return f.input
}

func (f *FullConnect) GetOutput() data.Data {
	return f.output
}

func (f *FullConnect) GetErrorToOutput() data.Data {
	return f.errorToOutput
}

func (f *FullConnect) GetOutputToInput() data.Data {
	return f.outputToInput
}

func (f *FullConnect) GetInputSize() []int {
	return nil
}

func (f *FullConnect) GetOutputSize() []int {
	return []int{f.Size}
}
