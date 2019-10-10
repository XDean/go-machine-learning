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

func (m *Model) Feed(input, output Data) {
	m.Forward(input)
	err := m.CalcError(output)
	m.Backward(err, output)
	m.Learn()
}

func (m *Model) Test(input, output Data) float64 {
	m.Forward(input)
	return m.CalcError(output)
}

func (m *Model) Predict(input Data) Data {
	m.Forward(input)
	return m.lastLayer().GetOutput()
}

func (m *Model) Forward(input Data) {
	m.Layers[0].Forward()
}

func (m *Model) CalcError(target Data) float64 {
	actual := m.lastLayer().GetOutput()
	return m.ErrorFunc(target, actual)
}

func (m *Model) Backward(error float64, target Data) {
	m.lastLayer().Backward(error, target)
}

func (m *Model) Learn() {
	m.lastLayer().Learn()
}

// Private
func (m *Model) lastLayer() Layer {
	return m.Layers[len(m.Layers)-1]
}
