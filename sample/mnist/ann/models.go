package main

import (
	"github.com/XDean/go-machine-learning/ann/classic/activation"
	"github.com/XDean/go-machine-learning/ann/classic/layer"
	"github.com/XDean/go-machine-learning/ann/classic/loss"
	"github.com/XDean/go-machine-learning/ann/classic/weight"
	"github.com/XDean/go-machine-learning/ann/core/model"
)

func init() {
	// 92.94%
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}
	layer.FullConnectDefaultConfig.LearningRatio = 0.1
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	RegisterModel(&model.Model{
		Name:      "DNN - Sigmoid - Square Loss - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.Square{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})

	// 95.08%
	layer.FullConnectDefaultConfig.Activation = activation.ReLU{}
	layer.FullConnectDefaultConfig.LearningRatio = 0.1
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.01}
	RegisterModel(&model.Model{
		Name:      "DNN - ReLU - SoftMax - CrossEntropy - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
			layer.NewSoftMax(),
		},
	})

	layer.ConvolutionDefaultConfig.LearningRatio = 0.1
	layer.ConvolutionDefaultConfig.WeightInit = &weight.RandomInit{Range: 0.01}
	layer.ConvolutionDefaultConfig.Activation = activation.ReLU{}

	layer.FullConnectDefaultConfig.LearningRatio = 0.1
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}

	// 93.46%
	RegisterModel(&model.Model{
		Name:      "CNN AS DNN - Sigmoid - Square Loss - (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
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

	layer.ConvolutionDefaultConfig.LearningRatio = 0.1
	layer.ConvolutionDefaultConfig.WeightInit = &weight.NormalInit{Std: 1}
	layer.ConvolutionDefaultConfig.Activation = activation.Sigmoid{}

	layer.FullConnectDefaultConfig.LearningRatio = 0.1
	layer.FullConnectDefaultConfig.WeightInit = &weight.RandomInit{Range: 1}
	layer.FullConnectDefaultConfig.Activation = activation.Sigmoid{}

	RegisterModel(&model.Model{
		Name:      "CNN - LeNet-5",
		ErrorFunc: loss.Square{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 6,
				KernelSize:  5,
				Padding:     2,
			}), // 6 * 28 * 28
			layer.NewPooling(layer.PoolingConfig{
				Type:    layer.POOL_MAX,
				Size:    1,
				Stride:  1,
				Padding: 0,
			}), // 6 * 14 * 14
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 16,
				KernelSize:  5,
				Padding:     0,
			}), // 16 * 10 * 10
			layer.NewPooling(layer.PoolingConfig{
				Type:    layer.POOL_MAX,
				Size:    1,
				Stride:  1,
				Padding: 0,
			}), // 6 * 5 * 5
			layer.NewFullConnect(layer.FullConnectConfig{Size: 120}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 84}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})
}
