package model

import (
	"github.com/XDean/go-machine-learning/ann/core/data"
)

type StartLayer struct {
	BaseLayer
	Input data.Data
}

func NewStartLayer(input data.Data) *StartLayer {
	return &StartLayer{Input: input}
}

func (s *StartLayer) Name() string {
	return "Start Layer"
}

func (s *StartLayer) Init() {
	// do nothing
}

func (s *StartLayer) Forward() {
	// do nothing
}

func (s *StartLayer) Backward() {
	// do nothing
}

func (s *StartLayer) Learn() {
	// do nothing
}

func (s *StartLayer) GetInput() data.Data {
	return s.Input
}

func (s *StartLayer) GetOutput() data.Data {
	return s.Input
}

func (s *StartLayer) GetErrorToInput() data.Data {
	return data.NewData()
}

func (s *StartLayer) GetOutputSize() []int {
	return s.Input.GetSize()
}

func (s *StartLayer) SetPrev(l Layer) {
	panic("start layer must has no prev")
}
