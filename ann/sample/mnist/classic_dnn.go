package main

import (
	"github.com/XDean/go-machine-learning/ann/classic"
	"github.com/XDean/go-machine-learning/ann/model"
)

func init() {
	//classic.FullLayerDefaultConfig.Activation = classic.ReLU{}
	//classic.FullLayerDefaultConfig.WeightInit = &classic.RandomInit{PositiveOnly: true, Range: 0.1}
	RegisterModel(&model.Model{
		Name:      "Classic DNN (28 * 28) * 200 * 40 * 10",
		ErrorFunc: classic.SquareError{},
		InputSize: []uint{28, 28},
		Layers: []model.Layer{
			classic.NewFullLayer(classic.FullLayerConfig{Size: 200}),
			classic.NewFullLayer(classic.FullLayerConfig{Size: 40}),
			classic.NewFullLayer(classic.FullLayerConfig{Size: 10}),
		},
	})
}
