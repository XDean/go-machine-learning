package main

import (
	"github.com/XDean/go-machine-learning/ann/classic"
	"github.com/XDean/go-machine-learning/ann/classic/layer"
	"github.com/XDean/go-machine-learning/ann/core/model"
)

func init() {
	RegisterModel(&model.Model{
		Name:      "Classic DNN (28 * 28) * 200 * 40 * 10",
		ErrorFunc: classic.SquareError{},
		InputSize: []int{28, 28},
		Layers: []model.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})
}
