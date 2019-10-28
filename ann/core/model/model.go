package model

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/core/util"
	"io"
	"os"
	"path/filepath"
	"time"
)

type (
	Model struct {
		Name      string
		Layers    []Layer
		ErrorFunc LossFunc
		InputSize [3]int
	}

	modelContext struct {
		model    *Model
		start    *StartLayer
		end      *EndLayer
		contexts []Context
	}

	Result struct {
		Output     Data
		Target     Data
		TotalError float64
		Time       time.Duration
	}
)

func (m *Model) Init() {
	start := m.newStart(NewData(m.InputSize))
	end := m.newEnd(NewData(m.lastLayer().GetOutputSize()))
	switch len(m.Layers) {
	case 0:
		panic("No layer in model")
	case 1:
		m.Layers[0].Init(start, end)
	default:
		for i, l := range m.Layers {
			if i == 0 {
				l.Init(start, m.Layers[i+1])
			} else if i == len(m.Layers)-1 {
				l.Init(m.Layers[i-1], end)
			} else {
				l.Init(m.Layers[i-1], m.Layers[i+1])
			}
		}
	}
}

func (m *Model) FeedTimes(input, target Data, times int) []Result {
	result := make([]Result, times)
	for i := range result {
		result[i] = m.Feed(input, target)
	}
	return result
}

func (m *Model) Feed(input, target Data) Result {
	startTime := time.Now()

	c := modelContext{
		model:    m,
		start:    m.newStart(input),
		end:      m.newEnd(target),
		contexts: make([]Context, len(m.Layers)),
	}
	c.Init()
	c.forward()
	c.backward()
	c.learn()

	return c.end.ToResult(startTime)
}

func (m *Model) Test(input, target Data) Result {
	startTime := time.Now()

	c := modelContext{
		model:    m,
		start:    m.newStart(input),
		end:      m.newEnd(target),
		contexts: make([]Context, len(m.Layers)),
	}
	c.Init()
	c.forward()

	return c.end.ToResult(startTime)
}

func (m *Model) Predict(input Data) Data {
	c := modelContext{
		model:    m,
		start:    m.newStart(input),
		end:      m.newEnd(EMPTY_DATA),
		contexts: make([]Context, len(m.Layers)),
	}
	c.Init()
	c.forward()
	return c.lastContext().GetOutput()
}

// Persistent
func (m *Model) SaveToFile(file string) (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(os.MkdirAll(filepath.Dir(file), 0755))
	writer, err := os.Create(file)
	util.NoError(err)
	defer writer.Close()
	return m.Save(writer)
}

func (m *Model) Save(writer io.Writer) (err error) {
	defer util.RecoverNoError(&err)
	encoder := gob.NewEncoder(writer)
	util.NoError(encoder.Encode(m))
	return nil
}

func (m *Model) LoadFromFile(file string) (err error) {
	defer util.RecoverNoError(&err)
	reader, err := os.Open(file)
	util.NoError(err)
	defer reader.Close()
	return m.Load(reader)
}

func (m *Model) Load(reader io.Reader) (err error) {
	defer util.RecoverNoError(&err)
	decoder := gob.NewDecoder(reader)
	util.NoError(decoder.Decode(m))
	return nil
}

// Private
func (c *modelContext) Init() {
	for i := range c.contexts {
		c.contexts[i] = c.model.Layers[i].NewContext()
	}
}

func (c *modelContext) forward() {
	c.start.Forward(nil)
	for i, ctx := range c.contexts {
		if i == 0 {
			ctx.Forward(c.start)
		} else {
			ctx.Forward(c.contexts[i-1])
		}
	}
	c.end.Forward(c.lastContext())
}

func (c *modelContext) backward() {
	c.end.Backward(c.contexts[0])
	layerSize := len(c.contexts)
	for i := layerSize; i > 0; i-- {
		ctx := c.contexts[i-1]
		if i == layerSize {
			ctx.Backward(c.end)
		} else {
			ctx.Backward(c.contexts[i])
		}
	}
	c.start.Backward(nil)
}

func (c *modelContext) learn() {
	for i := len(c.model.Layers) - 1; i >= 0; i-- {
		layer := c.model.Layers[i]
		layer.Learn(c.contexts[i : i+1])
	}
}

func (c *modelContext) lastContext() Context {
	return c.contexts[len(c.contexts)-1]
}

func (m *Model) lastLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *Model) newStart(input Data) *StartLayer {
	return NewStartLayer(input)
}

func (m *Model) newEnd(target Data) *EndLayer {
	return NewEndLayer(m.ErrorFunc, target)
}
