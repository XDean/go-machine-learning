package layer

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
	"math"
)

func init() {
	persistent.Register(new(SoftMax))
}

type SoftMax struct {
	BaseLayer

	Size Size

	output        Data
	errorToOutput Data
}

func NewSoftMax() *SoftMax {
	return &SoftMax{}
}

func (s *SoftMax) Init() {
	inputSize := s.GetPrev().GetOutputSize()
	if !s.BaseLayer.Init {
		s.BaseLayer.Init = true
		s.Size = inputSize
	}
	s.output = NewData(s.Size)
}

func (s *SoftMax) Forward() {
	input := s.GetPrev().GetOutput()
	max := 0.0
	input.ForEach(func(value float64) {
		if value > max {
			max = value
		}
	})
	sum := 0.0
	input.ForEachIndex(func(i, j, k int, value float64) {
		exp := math.Exp(value - max)
		sum += exp
		s.output.Value[i][j][k] = exp
	})
	s.output.Map(func(value float64) float64 {
		return value / sum
	})
}

func (s *SoftMax) Backward() {
	s.errorToOutput = s.GetNext().GetErrorToInput()
}

func (s *SoftMax) Learn() {
	// do nothing
}

func (s *SoftMax) GetOutput() Data {
	return s.output
}

func (s *SoftMax) GetErrorToInput() Data {
	result := NewData(s.Size)
	result.MapIndex(func(i1, j1, k1 int, _ float64) float64 {
		sum := 0.0
		s.errorToOutput.ForEachIndex(func(i2, j2, k2 int, value float64) {
			outputValue := s.output.Value[i2][j2][k2]
			if i1 == i2 && j1 == j2 && k1 == k2 {
				sum += value * outputValue * (1 - outputValue)
			} else {
				sum += value * -outputValue * s.output.Value[i1][j1][k1]
			}
		})
		return sum
	})
	return result
}

func (s *SoftMax) GetOutputSize() Size {
	return s.Size
}
