package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/XDean/go-machine-learning/ann/weight"
)

func init() {
	persistent.Register(new(C3))
}

var C3Config = struct {
	D1, K, F, S, P int
	Table          [6][16]int
}{
	D1: 6,
	K:  16,
	F:  5,
	S:  1,
	P:  0,
	Table: [6][16]int{
		{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
		{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
		{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
		{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
		{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
		{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1},
	},
}

type (
	C3 struct {
		WeightInit    weight.Init
		WeightFactory weight.Factory

		WeightSize core.Size
		Weight     [][][][]weight.Weight // K * F * F * D1
		Bias       []weight.Weight       // K

		InputSize core.Size // D1 * W1 * H1
		// W2 = (W1 + 2P - F) / S + 1
		// H2 = (H1 + 2P - F) / S + 1
		// D2 = K
		OutputSize core.Size // K * W2 * H2
	}

	c3Context struct {
		layer         *C3
		input         core.Data   // D1 * W1 * H1
		output        core.Data   // K * W2 * H2
		errorToOutput core.Data   // output, ∂E / ∂a
		errorToWeight []core.Data // weight, ∂E / ∂w
		errorToBias   []float64
		outputToNet   core.Data
		netToInput    [][][]core.Data // output * F * F * D1, ∂a / ∂i, no active partial
		netToWeight   [][][]core.Data // output * F * F * D1, ∂a / ∂w, no active partial
	}
)

func (f *C3) Init(prev, next core.Layer) {
	inputSize := prev.GetOutputSize()
	f.WeightSize = [3]int{inputSize[0], C3Config.F, C3Config.F}
	f.InputSize = inputSize
	f.OutputSize = [3]int{
		C3Config.K,
		(inputSize[1]+2*C3Config.P-C3Config.F)/C3Config.S + 1,
		(inputSize[2]+2*C3Config.P-C3Config.F)/C3Config.S + 1,
	}
	f.Weight = make([][][][]weight.Weight, C3Config.K)
	for i := range f.Weight {
		f.Weight[i] = weight.Create3D(f.WeightFactory, f.WeightInit, f.WeightSize)
	}
	f.Bias = weight.Create1D(f.WeightFactory, f.WeightInit, C3Config.K)
}

func (f *C3) Learn(ctxs []core.Context) {
	size := float64(len(ctxs))
	for n := range f.Weight {
		for i := range f.Weight[n] {
			for j := range f.Weight[n][i] {
				for k := range f.Weight[n][i][j] {
					gradient := 0.0
					for _, v := range ctxs {
						ctx := v.(*c3Context)
						gradient += ctx.errorToWeight[n].Value[i][j][k] / size
					}
					f.Weight[n][i][j][k].Learn(gradient)
				}
			}
		}
	}
	for i := range f.Bias {
		gradient := 0.0
		for _, v := range ctxs {
			ctx := v.(*c3Context)
			gradient += ctx.errorToBias[i] / size
		}
		f.Bias[i].Learn(gradient)
	}
}

func (f *C3) NewContext() core.Context {
	errorToWeight := make([]core.Data, C3Config.K)
	weightSize := [3]int{f.InputSize[0], C3Config.F, C3Config.F}
	for i := range errorToWeight {
		errorToWeight[i] = core.NewData(weightSize)
	}
	newO2W := func() [][][]core.Data {
		result := make([][][]core.Data, f.OutputSize[0])
		for i := range result {
			result[i] = make([][]core.Data, f.OutputSize[1])
			for j := range result[i] {
				result[i][j] = make([]core.Data, f.OutputSize[2])
				for k := range result[i][j] {
					result[i][j][k] = core.NewData(f.WeightSize)
				}
			}
		}
		return result
	}
	return &c3Context{
		layer:         f,
		output:        core.NewData(f.OutputSize),
		errorToOutput: core.NewData(f.OutputSize),
		errorToWeight: errorToWeight,
		errorToBias:   make([]float64, C3Config.K),
		netToWeight:   newO2W(),
		netToInput:    newO2W(),
		outputToNet:   core.NewData(f.OutputSize),
	}
}

func (f *c3Context) Forward(prev core.Context) {
	f.input = prev.GetOutput()
	f.output.ForEachIndex(func(kernel, x, y int, value float64) {
		net := f.layer.Bias[kernel].Get()
		for z := 0; z < f.layer.InputSize[0]; z++ {
			if C3Config.Table[z][kernel] == 0 {
				continue
			}
			for i := 0; i < C3Config.F; i++ {
				for j := 0; j < C3Config.F; j++ {
					inputX := x + i - C3Config.P
					inputY := y + j - C3Config.P
					w := f.layer.Weight[kernel][z][i][j].Get()
					isPadding := inputX < 0 || inputX >= f.layer.InputSize[1] || inputY < 0 || inputY >= f.layer.InputSize[2]
					inputValue := 0.0
					if !isPadding {
						inputValue = f.input.Value[z][inputX][inputY]
					}
					net += inputValue * w
					if !isPadding {
						f.netToInput[kernel][x][y].Value[z][i][j] = w
						f.netToWeight[kernel][x][y].Value[z][i][j] = inputValue
					}
				}
			}
		}
		output, partial := Activation{}.Active(net)
		f.output.Value[kernel][x][y] = output
		f.outputToNet.Value[kernel][x][y] = partial
	})
}

func (f *c3Context) Backward(next core.Context) {
	f.errorToOutput = next.GetErrorToInput()
	for kernel, v := range f.errorToWeight {
		v.MapIndex(func(i, j, k int, value float64) float64 {
			sum := 0.0
			for x := range f.errorToOutput.Value[kernel] {
				for y := range f.errorToOutput.Value[kernel][x] {
					sum += f.errorToOutput.Value[kernel][x][y] * f.outputToNet.Value[kernel][x][y] * f.netToWeight[kernel][x][y].Value[i][j][k]
				}
			}
			return sum
		})
		sum := 0.0
		for x := range f.errorToOutput.Value[kernel] {
			for y := range f.errorToOutput.Value[kernel][x] {
				sum += f.errorToOutput.Value[kernel][x][y] * f.outputToNet.Value[kernel][x][y]
			}
		}
		f.errorToBias[kernel] = sum
	}
}

func (f *c3Context) GetOutput() core.Data {
	return f.output
}

func (f *c3Context) GetErrorToInput() core.Data {
	result := core.NewData(f.layer.InputSize)
	f.errorToOutput.ForEachIndex(func(kernel, x, y int, v float64) {
		for z := 0; z < f.layer.InputSize[0]; z++ {
			if C3Config.Table[z][kernel] == 0 {
				continue
			}
			for i := 0; i < C3Config.F; i++ {
				for j := 0; j < C3Config.F; j++ {
					inputX := x + i - C3Config.P
					inputY := y + j - C3Config.P
					isPadding := inputX < 0 || inputX >= f.layer.InputSize[1] || inputY < 0 || inputY >= f.layer.InputSize[2]
					if isPadding {
						continue
					}
					result.Value[z][inputX][inputY] += v * f.netToInput[kernel][x][y].Value[z][i][j] * f.outputToNet.Value[kernel][x][y]
				}
			}
		}
	})
	return result
}

func (f *C3) GetOutputSize() core.Size {
	return f.OutputSize
}

func (f *C3) Desc() core.Desc {
	return core.SimpleDesc{
		Name: "Convolution",
		Core: f.OutputSize,
		Params: map[string]interface{}{
			"Kernel":     C3Config.K,
			"Size":       C3Config.F,
			"Stride":     C3Config.S,
			"Padding":    C3Config.P,
			"Activation": Activation{},
			"Weight":     f.WeightFactory,
		},
	}
}
