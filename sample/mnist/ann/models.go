package main

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	core2 "github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/layer"
	"github.com/XDean/go-machine-learning/ann/loss"
	"github.com/XDean/go-machine-learning/ann/weight"
)

func init() {
	// 92.94%
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}
	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	RegisterModel(&core2.Model{
		Name:      "DNN - Sigmoid - Square Loss - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.Square{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core2.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})

	// 95.08%
	layer.FullConnectDefaultConfig.Activation = activation.ReLU{}
	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.1}
	RegisterModel(&core2.Model{
		Name:      "DNN - ReLU - SoftMax - CrossEntropy - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core2.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10, Activation: activation.NoOp{}}),
			layer.NewSoftMax(),
		},
	})

	layer.ConvolutionDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.ConvolutionDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.01}
	layer.ConvolutionDefaultConfig.Activation = activation.ReLU{}

	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}

	// 93.46%
	RegisterModel(&core2.Model{
		Name:      "CNN AS DNN - ReLU - SoftMax - CrossEntropy - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core2.Layer{
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 200,
				KernelSize:  28,
				Padding:     0,
			}), // 200
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 40,
				KernelSize:  1,
				Padding:     0,
			}), // 40
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
			layer.NewSoftMax(),
		},
	})

	layer.ConvolutionDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.ConvolutionDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.ConvolutionDefaultConfig.Activation = activation.Tanh{}

	layer.FullConnectDefaultConfig.WeightFactory = weight.SGDFactory{Eta: 0.1}
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.FullConnectDefaultConfig.Activation = activation.Tanh{}

	RegisterModel(&core2.Model{
		Name:      "CNN - LeNet-5",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []core2.Layer{
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 6,
				KernelSize:  5,
				Padding:     2,
			}), // 6 * 28 * 28
			layer.NewPooling(layer.PoolingConfig{
				Type:    layer.POOL_SUM,
				Size:    2,
				Stride:  2,
				Padding: 0,
			}), // 6 * 14 * 14
			layer.NewActivation(activation.Sigmoid{}),
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 16,
				KernelSize:  5,
				Padding:     0,
			}), // 16 * 10 * 10
			layer.NewPooling(layer.PoolingConfig{
				Type:    layer.POOL_SUM,
				Size:    2,
				Stride:  2,
				Padding: 0,
			}), // 6 * 5 * 5
			layer.NewActivation(activation.Sigmoid{}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 120}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 84}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10, Activation: activation.NoOp{}}),
			layer.NewSoftMax(),
		},
	})
}
