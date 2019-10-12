package classic

import (
	"encoding/gob"
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"math/rand"
)

type WeightInit interface {
	persistent.Persistent
	Init(data base.Data) base.Data
}

type RandomInit struct {
	PositiveOnly bool
	Range        float64
}

func (r *RandomInit) Name() string {
	return "WeightInit-RandomInit"
}

func (r *RandomInit) Init(data base.Data) base.Data {
	data.ForEach(func(index []uint, value float64) {
		v := rand.Float64()
		if !r.PositiveOnly {
			v = (v - 0.5) * 2
		}
		v *= r.Range
		data.SetValue(v, index...)
	})
	return data
}

func (r *RandomInit) Save(writer *gob.Encoder) error {
	return writer.Encode(r)
}

func (r *RandomInit) Load(reader *gob.Decoder) error {
	return reader.Decode(&r)
}
