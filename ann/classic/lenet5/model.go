package lenet5

import (
	"github.com/XDean/go-machine-learning/ann/activation"
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/layer"
	"github.com/XDean/go-machine-learning/ann/loss"
	"github.com/XDean/go-machine-learning/ann/weight"
)

var Model = &core.Model{
	Name:      "LeNet-5",
	ErrorFunc: loss.CrossEntropy{},
	InputSize: [3]int{1, 28, 28},
	Layers: []core.Layer{
		layer.NewConvolution(layer.ConvolutionConfig{
			KernelCount: 6,
			KernelSize:  5,
			Padding:     2,
		}), // 6 * 28 * 28
		&SubSampling{
			Size:    2,
			Stride:  2,
			Padding: 0,
		}, // 6 * 14 * 14
		layer.NewActivation(activation.Sigmoid{}),
		&C3{
			WeightInit: weight.RandomInit{Range: 1},
		}, // 16 * 10 * 10
		&SubSampling{
			Size:    2,
			Stride:  2,
			Padding: 0,
		}, // 16 * 5 * 5
		layer.NewActivation(activation.Sigmoid{}),
		layer.NewConvolution(layer.ConvolutionConfig{
			KernelCount: 120,
			KernelSize:  5,
			Padding:     0,
		}), // 120 * 1 * 1
		layer.NewFullConnect(layer.FullConnectConfig{
			Size:       84,
			Activation: Activation{},
		}),
		NewRBF(),
	},
}
