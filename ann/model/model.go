package model

type Model struct {
	Layers []Layer
	Error  float64
}

func (m *Model) Feed(input, output *Data) {
	m.Forward(input)
	m.CalcError(output)
	m.Backward(output)
	m.Learn()
}

func (m *Model) Test(input, output *Data) float64 {
	m.Forward(input)
	m.CalcError(output)
	return m.Error
}

func (m *Model) Predict(input *Data) *Data {
	m.Forward(input)
	return m.Layers[len(m.Layers)-1].GetOutput()
}

func (m *Model) Forward(input *Data) {
}

func (m *Model) CalcError(target *Data) {
}

func (m *Model) Backward(target *Data) {
}

func (m *Model) Learn() {
}
