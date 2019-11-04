package main

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/classic/lenet5"
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/layer"
	"github.com/XDean/go-machine-learning/ann/loss"
	"github.com/XDean/go-machine-learning/ann/weight"
	. "github.com/XDean/go-machine-learning/sample/mnist/ann"
)

func registerModels() {
	// 92.94%
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}
	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	RegisterModel(&core.Model{
		ErrorFunc: loss.Square{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	}, MaxLabel)

	// 95.08%
	layer.FullConnectDefaultConfig.Activation = activation.ReLU{}
	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.1}
	RegisterModel(&core.Model{
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10, Activation: activation.NoOp{}}),
			layer.NewSoftMax(),
		},
	}, MaxLabel)

	layer.ConvolutionDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.ConvolutionDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.01}
	layer.ConvolutionDefaultConfig.Activation = activation.ReLU{}

	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}

	RegisterModel(lenet5.Model, MinLabel)
}
