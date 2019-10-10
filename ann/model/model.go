package model

type (
	Model struct {
		Layers    []Layer
		ErrorFunc ErrorFunc
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

func (m *Model) Feed(input, target Data) {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	end.Backward()
	end.Learn()
}

func (m *Model) Test(input, target Data) float64 {
	start := m.newStart(input)
	end := m.newEnd(target)

	start.Forward()
	return end.TotalError
}

func (m *Model) Predict(input Data) Data {
	start := m.newStart(input)
	start.Forward()
	return m.lastLayer().GetOutput()
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

// Private
func (m *Model) lastLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}
