package model

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
)

type StartLayer struct {
	BaseLayer
	Input base.Data
}

func NewStartLayer(input base.Data) *StartLayer {
	return &StartLayer{Input: input}
}

func (s *StartLayer) Name() string {
	return "Start Layer"
}

func (s *StartLayer) Save(writer *gob.Encoder) error {
	panic("no save")
}

func (s *StartLayer) Load(reader *gob.Decoder) error {
	panic("no load")
}

func (s *StartLayer) Forward() {
	s.Next.Forward()
}

func (s *StartLayer) Backward() {
	// do nothing
}

func (s *StartLayer) Learn() {
	// do nothing
}

func (s *StartLayer) GetInput() base.Data {
	return s.Input
}

func (s *StartLayer) GetOutput() base.Data {
	return s.Input
}

func (s *StartLayer) GetErrorToOutput() base.Data {
	return base.NewData()
}

func (s *StartLayer) GetOutputToInput() base.Data {
	return s.Input.Identity2D()
}

func (s *StartLayer) GetInputSize() []uint {
	return s.Input.GetSize()
}

func (s *StartLayer) GetOutputSize() []uint {
	return s.Input.GetSize()
}

func (s *StartLayer) SetPrev(l Layer) {
	panic("start layer must has no prev")
}
