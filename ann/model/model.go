package model

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/persistent"
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
		Output     base.Data
		Target     base.Data
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

func (m *Model) Feed(input, target base.Data) Result {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	end.Backward()
	end.Learn()

	return end.ToResult()
}

func (m *Model) Test(input, target base.Data) Result {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	return end.ToResult()
}

func (m *Model) Predict(input base.Data) base.Data {
	start := m.newStart(input)
	start.Forward()
	return m.lastLayer().GetOutput()
}

func (m *Model) SaveToFile(file string) (err error) {
	defer base.RecoverNoError(&err)
	base.NoError(os.MkdirAll(filepath.Dir(file), 0755))
	writer, err := os.Create(file)
	base.NoError(err)
	defer writer.Close()
	return m.Save(writer)
}

func (m *Model) Save(writer io.Writer) (err error) {
	defer base.RecoverNoError(&err)
	encoder := gob.NewEncoder(writer)
	base.NoError(encoder.Encode(len(m.Layers)))
	for _, v := range m.Layers {
		base.NoError(persistent.Save(encoder, v))
	}
	base.NoError(persistent.Save(encoder, m.ErrorFunc))
	return nil
}

func (m *Model) LoadFromFile(file string) (err error) {
	defer base.RecoverNoError(&err)
	reader, err := os.Open(file)
	base.NoError(err)
	defer reader.Close()
	return m.Load(reader)
}

func (m *Model) Load(reader io.Reader) (err error) {
	defer base.RecoverNoError(&err)
	decoder := gob.NewDecoder(reader)
	layers := make([]Layer, 0)
	count := 0
	base.NoError(decoder.Decode(&count))
	for ; count > 0; count-- {
		bean, err := persistent.Load(decoder)
		base.NoError(err)
		layer, ok := bean.(Layer)
		if !ok {
			return persistent.TypeError("Layer", bean)
		}
		base.NoError(bean.Load(decoder))
		layers = append(layers, layer)
	}
	bean, err := persistent.Load(decoder)
	base.NoError(err)

	errorFunc, ok := bean.(ErrorFunc)
	if !ok {
		return persistent.TypeError("ErrorFunc", bean)
	}
	m.Layers = layers
	m.ErrorFunc = errorFunc
	m.Init()
	return nil
}

// Private
func (m *Model) lastLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}

func (m *Model) newStart(input base.Data) *StartLayer {
	start := NewStartLayer(input)
	m.Layers[0].SetPrev(start)
	start.SetNext(m.Layers[0])
	return start
}

func (m *Model) newEnd(target base.Data) *EndLayer {
	end := NewEndLayer(m.ErrorFunc, target)
	end.SetPrev(m.lastLayer())
	m.lastLayer().SetNext(end)
	return end
}
