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
		ErrorFunc ErrorFunc
		InputSize [3]int
	}

	Result struct {
		Output     Data
		Target     Data
		TotalError float64
		Time       time.Duration
	}
)

func (m *Model) Init() {
	m.initLayer()
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

	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	m.forward()
	end.Forward()

	end.Backward()
	m.backward()
	start.Backward()

	end.Learn()
	m.learn()
	start.Learn()

	return end.ToResult(startTime)
}

func (m *Model) Test(input, target Data) Result {
	startTime := time.Now()

	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	m.forward()
	end.Forward()

	return end.ToResult(startTime)
}

func (m *Model) Predict(input Data) Data {
	start := m.newStart(input)
	start.Forward()
	m.forward()
	return m.lastLayer().GetOutput()
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

func (m *Model) forLayer(f func(int, Layer)) {
	for i, v := range m.Layers {
		f(i, v)
	}
}

func (m *Model) forLayerReverse(f func(int, Layer)) {
	for i := len(m.Layers); i > 0; i-- {
		f(i-1, m.Layers[i-1])
	}
}

func (m *Model) forward() {
	m.forLayer(func(i int, layer Layer) {
		layer.Forward()
		//fmt.Println(layer.GetOutput().ToArray())
	})
}

func (m *Model) backward() {
	m.forLayerReverse(func(i int, layer Layer) {
		layer.Backward()
	})
}

func (m *Model) learn() {
	m.forLayerReverse(func(i int, layer Layer) {
		layer.Learn()
	})
}

func (m *Model) initLayer() {
	start := m.newStart(NewData(m.InputSize))
	end := m.newEnd(NewData(m.lastLayer().GetOutputSize()))
	for i, l := range m.Layers {
		if i == 0 {
			l.SetPrev(start)
		} else {
			l.SetPrev(m.Layers[i-1])
		}
		if i == len(m.Layers)-1 {
			l.SetNext(end)
		} else {
			l.SetNext(m.Layers[i+1])
		}
		l.Init()
	}
}
