package core

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/util"
	"io"
	"os"
	"path/filepath"
	"sync"
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

	TrainData struct {
		Input, Target Data
	}

	Result struct {
		Output     Data
		Target     Data
		TotalError float64
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

func (m *Model) FeedBatch(dataStream <-chan TrainData, batchSize int) <-chan Result {
	result := make(chan Result, batchSize)
	go func() {
		batch := make([]*modelContext, batchSize)
		done := sync.WaitGroup{}
		for {
			for i := range batch {
				data, ok := <-dataStream
				if !ok {
					batch = batch[:i]
					break
				}
				done.Add(1)
				c := &modelContext{
					model: m,
					start: m.newStart(data.Input),
					end:   m.newEnd(data.Target),
				}
				batch[i] = c
				go func() {
					c.init()
					c.forward()
					c.backward()
					result <- c.end.ToResult()
					done.Done()
				}()
			}
			done.Wait()
			m.learn(batch)
			if len(batch) != batchSize {
				break
			}
		}
		close(result)
	}()
	return result
}

func (m *Model) Feed(input, target Data) Result {
	c := &modelContext{
		model: m,
		start: m.newStart(input),
		end:   m.newEnd(target),
	}
	c.init()
	c.forward()
	c.backward()
	m.learn([]*modelContext{c})

	return c.end.ToResult()
}

func (m *Model) Test(input, target Data) Result {
	c := modelContext{
		model: m,
		start: m.newStart(input),
		end:   m.newEnd(target),
	}
	c.init()
	c.forward()

	return c.end.ToResult()
}

func (m *Model) Predict(input Data) Data {
	c := modelContext{
		model: m,
		start: m.newStart(input),
		end:   m.newEnd(EMPTY_DATA),
	}
	c.init()
	c.forward()
	return c.end.GetOutput()
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
func (c *modelContext) init() {
	c.contexts = make([]Context, len(c.model.Layers))
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

func (m *Model) learn(ctxs []*modelContext) {
	for i := len(m.Layers) - 1; i >= 0; i-- {
		contexts := make([]Context, len(ctxs))
		for j, c := range ctxs {
			contexts[j] = c.contexts[i]
		}
		m.Layers[i].Learn(contexts)
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
