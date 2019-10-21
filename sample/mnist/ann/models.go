package main

import (
	"github.com/XDean/go-machine-learning/ann/classic/layer"
	"github.com/XDean/go-machine-learning/ann/classic/loss"
	"github.com/XDean/go-machine-learning/ann/core/model"
)

func init() {
	//layer.FullConnectDefaultConfig.WeightInit = &classic.ZeroInit{}
	//layer.FullConnectDefaultConfig.Activation = classic.ReLU{}
	RegisterModel(&model.Model{
		Name:      "Classic DNN (28 * 28) * 200 * 40 * 10",
		ErrorFunc: loss.CrossEntropy{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
			layer.NewSoftMax(),
		},
	})

	RegisterModel(&model.Model{
		Name:      "Classic CNN",
		ErrorFunc: loss.SquareError{},
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewConvolution(layer.ConvolutionConfig{
				KernelCount: 25,
				KernelSize:  3,
				Padding:     1,
			}), // 25 * 28 * 28
			layer.NewPooling(layer.PoolingConfig{
				Type:    layer.POOL_MAX,
				Size:    2,
				Stride:  2,
				Padding: 0,
			}), // 25 * 14 * 14
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})
}
