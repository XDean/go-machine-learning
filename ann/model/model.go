package model

import (
	"encoding/gob"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/model/persistent"
	"io"
	"os"
	"path/filepath"
)

type (
	Model struct {
		Layers    []Layer
		ErrorFunc ErrorFunc
	}

	Result struct {
		Output     Data
		Target     Data
		TotalError float64
	}
)

func (m *Model) Init() {
	for i, l := range m.Layers {
		if i > 0 {
			l.SetPrev(m.Layers[i-1])
		}
		if i < len(m.Layers)-1 {
			l.SetNext(m.Layers[i+1])
		}
	}
}

func (m *Model) Feed(input, target Data) Result {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	end.Backward()
	end.Learn()

	return end.ToResult()
}

func (m *Model) Test(input, target Data) Result {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	return end.ToResult()
}

func (m *Model) Predict(input Data) Data {
	start := m.newStart(input)
	start.Forward()
	return m.lastLayer().GetOutput()
}

func (m *Model) SaveToFile(file string) (err error) {
	defer RecoverNoError(&err)
	NoError(os.MkdirAll(filepath.Dir(file), 0755))
	writer, err := os.Create(file)
	NoError(err)
	defer writer.Close()
	return m.Save(writer)
}

func (m *Model) Save(writer io.Writer) (err error) {
	defer RecoverNoError(&err)
	encoder := gob.NewEncoder(writer)
	NoError(encoder.Encode(len(m.Layers)))
	for _, v := range m.Layers {
		NoError(encoder.Encode(v.Name()))
		NoError(v.Save(encoder))
	}
	return nil
}

func (m *Model) LoadFromFile(file string) (err error) {
	defer RecoverNoError(&err)
	reader, err := os.Open(file)
	NoError(err)
	defer reader.Close()
	return m.Load(reader)
}

func (m *Model) Load(reader io.Reader) (err error) {
	defer RecoverNoError(&err)
	decoder := gob.NewDecoder(reader)
	layers := make([]Layer, 0)
	count := 0
	NoError(decoder.Decode(&count))
	for ; count > 0; count-- {
		name := ""
		NoError(decoder.Decode(&name))
		bean, err := persistent.New(name)
		NoError(err)
		layer, ok := bean.(Layer)
		if !ok {
			return fmt.Errorf("Bad type: expect Layer, but %T", bean)
		}
		NoError(bean.Load(decoder))
		layers = append(layers, layer)
	}
	m.Layers = layers
	m.Init()
	return nil
}

// Private
func (m *Model) lastLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *Model) newStart(input Data) *StartLayer {
	start := NewStartLayer(input)
	m.Layers[0].SetPrev(start)
	start.SetNext(m.Layers[0])
	return start
}

func (m *Model) newEnd(target Data) *EndLayer {
	end := NewEndLayer(m.ErrorFunc, target)
	end.SetPrev(m.lastLayer())
	m.lastLayer().SetNext(end)
	return end
}
