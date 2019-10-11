package classic

import (
	"github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/ann/model/persistent"
	"math/rand"
)

type WeightInit interface {
	persistent.Persistent
	Init(data model.Data) model.Data
}

type RandomInit struct {
	persistent.TypePersistent
}

func (r RandomInit) Name() string {
	return "WeightInit-RandomInit"
}

func (r RandomInit) Init(data model.Data) model.Data {
	data.ForEach(func(index []uint, value float64) {
		data.SetValue(rand.Float64()-0.5, index...)
	})
	return data
}
