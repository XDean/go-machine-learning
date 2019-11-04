package mnist_ann

import "github.com/XDean/go-machine-learning/ann/core"

type model struct {
	model         *core.Model
	outputToLabel func(output core.Data) int
}

var models = make([]*core.Model, 0)
var labelMap = make(map[*core.Model]func(core.Data) int)

func RegisterModel(m *core.Model, outputToLabel func(output core.Data) int) {
	models = append(models, m)
	labelMap[m] = outputToLabel
	m.Init()
}
