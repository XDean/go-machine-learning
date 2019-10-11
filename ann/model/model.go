package model

import (
	"encoding/gob"
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

func (m *Model) SaveToFile(file string) error {
	err := os.MkdirAll(filepath.Dir(file), 0755)
	if err != nil {
		return err
	}
	writer, err := os.Create(file)
	if err != nil {
		return err
	}
	defer writer.Close()
	return m.Save(writer)
}

func (m *Model) Save(writer io.Writer) error {
	//encoder := gob.NewEncoder(writer)
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
