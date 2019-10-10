package classic

import (
	"github.com/XDean/go-machine-learning/ann/model"
	"math/rand"
)

type WeightInit func(data model.Data) model.Data

func RandomInit() WeightInit {
	return func(data model.Data) model.Data {
		data.ForEach(func(index []uint, value float64) {
			data.SetValue(rand.Float64()-0.5, index...)
		})
		return data
	}
}
