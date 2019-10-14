package model

import (
	"github.com/XDean/go-machine-learning/ann/data"
	"github.com/XDean/go-machine-learning/ann/persistent"
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

func (s *StartLayer) Save(writer persistent.Encoder) error {
	panic("no save")
}

func (s *StartLayer) Load(reader persistent.Decoder) error {
	panic("no load")
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

func (s *StartLayer) GetErrorToOutput() data.Data {
	return data.NewData()
}

func (s *StartLayer) GetOutputToInput() data.Data {
	return data.Identity2D(s.Input)
}

func (s *StartLayer) GetInputSize() []int {
	return s.Input.GetSize()
}

func (s *StartLayer) GetOutputSize() []int {
	return s.Input.GetSize()
}

func (s *StartLayer) SetPrev(l Layer) {
	panic("start layer must has no prev")
}
