package layer

//
//import (
//	"github.com/XDean/go-machine-learning/ann/core/data"
//	. "github.com/XDean/go-machine-learning/ann/core/model"
//	"github.com/XDean/go-machine-learning/ann/core/persistent"
//	"github.com/XDean/go-machine-learning/ann/core/util"
//)
//
//func init() {
//	persistent.Register(new(Pooling))
//}
//
//type (
//	PoolingType int
//	Pooling     struct {
//		BaseLayer
//
//		Type    PoolingType
//		Size    int
//		Padding int
//		Stride  int
//
//		InputSize  [3]int
//		OutputSize [3]int
//
//		input             data.Data
//		output            data.Data
//		errorToOutput     data.Data                     // output
//		errorToInput      data.Data                     // input
//		outputToInput     map[[3]int]float64            // output
//		outputToEachInput map[[3]int]map[[3]int]float64 // output * input
//	}
//
//	PoolingConfig struct {
//		Type    PoolingType
//		Size    int
//		Padding int
//		Stride  int
//	}
//)
//
//const (
//	POOL_MAX PoolingType = iota + 1
//	POOL_AVG
//	POOL_SUM
//)
//
//var (
//	PoolingDefaultConfig = PoolingConfig{
//		Type:    POOL_MAX,
//		Size:    2,
//		Padding: 0,
//		Stride:  1,
//	}
//)
//
//func NewPooling(config PoolingConfig) *Pooling {
//	if config.Type == 0 {
//		config.Type = PoolingDefaultConfig.Type
//	}
//	if config.Size == 0 {
//		config.Size = PoolingDefaultConfig.Size
//	}
//	if config.Stride == 0 {
//		config.Stride = PoolingDefaultConfig.Stride
//	}
//	return &Pooling{
//		Type:    config.Type,
//		Size:    config.Size,
//		Stride:  config.Stride,
//		Padding: config.Padding,
//	}
//}
//
//func (f *Pooling) Init() {
//	inputSize := f.GetPrev().GetOutputSize()
//	util.MustTrue(len(inputSize) == 3)
//	f.InputSize = [3]int{inputSize[0], inputSize[1], inputSize[2]}
//	f.OutputSize = [3]int{
//		(f.InputSize[0]+2*f.Padding-f.Size)/f.Stride + 1,
//		(f.InputSize[1]+2*f.Padding-f.Size)/f.Stride + 1,
//		f.InputSize[2],
//	}
//	f.output = data.NewData(f.OutputSize[:])
//	f.errorToOutput = data.NewData(f.OutputSize[:])
//	f.outputToInput = make(map[[3]int]float64)
//	f.outputToEachInput = make(map[[3]int]map[[3]int]float64)
//}
//
//func (f *Pooling) Forward() {
//	f.input = f.GetPrev().GetOutput()
//
//	f.output.MapIndex(func(outputIndex []int, _ float64) float64 {
//		outputIndexArray := [3]int{}
//		copy(outputIndexArray[:], outputIndex)
//		f.outputToEachInput[outputIndexArray] = make(map[[3]int]float64)
//
//		depth := outputIndex[2]
//		maxIndex := []int{0, 0}
//		maxValue := 0.0
//		sum := 0.0
//		for i := 0; i < f.Size; i++ {
//			for j := 0; j < f.Size; j++ {
//				inputX := outputIndex[0] + i - f.Padding
//				inputY := outputIndex[1] + j - f.Padding
//				inputValue := func() (result float64) {
//					if inputX < 0 || inputX >= f.InputSize[0] || inputY < 0 || inputY >= f.InputSize[1] {
//						return 0.0
//					} else if f.input.GetDim() == 2 {
//						return f.input.GetValue([]int{inputX, inputY})
//					} else {
//						return f.input.GetValue([]int{inputX, inputY, depth})
//					}
//				}()
//				switch f.Type {
//				case POOL_MAX:
//					if inputValue > maxValue {
//						maxValue = inputValue
//						maxIndex[0] = i
//						maxIndex[1] = j
//					}
//				case POOL_AVG:
//					f.outputToEachInput[outputIndexArray][[3]int{i, j, depth}] = 1 / float64(f.Size*f.Size)
//					sum += inputValue
//				case POOL_SUM:
//					f.outputToEachInput[outputIndexArray][[3]int{i, j, depth}] = 1
//					sum += inputValue
//				}
//			}
//		}
//		switch f.Type {
//		case POOL_MAX:
//			f.outputToEachInput[outputIndexArray][[3]int{maxIndex[0], maxIndex[1], depth}] = 1
//			return maxValue
//		case POOL_AVG:
//			return sum / float64(f.Size*f.Size)
//		case POOL_SUM:
//			return sum
//		}
//		return 0
//	})
//}
//
//func (f *Pooling) Backward() {
//	f.errorToOutput = f.GetNext().GetErrorToInput()
//	outputIndexArray := [3]int{}
//	inputIndexArray := [3]int{}
//	f.errorToInput = ErrorToInputByFunc(f.errorToOutput, f.InputSize[:], func(outputIndex, inputIndex []int) float64 {
//		copy(outputIndexArray[:], outputIndex)
//		copy(inputIndexArray[:], inputIndex)
//		return f.outputToEachInput[outputIndexArray][inputIndexArray]
//	})
//}
//
//func (f *Pooling) Learn() {
//	// do nothing
//}
//
//func (f *Pooling) GetInput() data.Data {
//	return f.input
//}
//
//func (f *Pooling) GetOutput() data.Data {
//	return f.output
//}
//
//func (f *Pooling) GetErrorToInput() data.Data {
//	return f.errorToInput
//}
//
//func (f *Pooling) GetOutputSize() []int {
//	return f.OutputSize[:]
//}
