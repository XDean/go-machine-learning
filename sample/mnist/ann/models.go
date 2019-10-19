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
		InputSize: [3]int{1, 28, 28},
		Layers: []model.Layer{
			layer.NewFullConnect(layer.FullConnectConfig{Size: 200}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 40}),
			layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
		},
	})

	//RegisterModel(&model.Model{
	//	Name:      "Classic CNN",
	//	ErrorFunc: classic.SquareError{},
	//	InputSize: []int{28, 28},
	//	Layers: []model.Layer{
	//		layer.NewDimAdd(1),
	//		layer.NewConvolution(layer.ConvolutionConfig{
	//			KernelCount: 10,
	//			KernelSize:  3,
	//			Padding:     1,
	//		}), // 28 * 28 * 10
	//		layer.NewPooling(layer.PoolingConfig{
	//			Type: layer.POOL_MAX,
	//			Size: 2,
	//		}), // 27 * 27 * 10
	//		layer.NewFullConnect(layer.FullConnectConfig{Size: 10}),
	//	},
	//})
}
