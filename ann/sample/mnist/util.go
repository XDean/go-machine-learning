package main

import (
	"github.com/XDean/go-machine-learning/ann/data"
	"github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/data/mnist"
)

func mnistToData(d mnist.MnistData) (input, target data.Data) {
	input = data.NewData(28, 28)
	target = data.NewData(10)

	input.ForEach(func(index []int, value float64) {
		input.SetValue(float64(d.Image[index[0]*28+index[1]])/255.0, index...)
	})
	target.SetValue(1, int(d.Label))
	return
}

func predictFromResult(r model.Result) int {
	max := int(0)
	r.Output.ForEach(func(index []int, value float64) {
		if value > r.Output.GetValue(max) {
			max = index[0]
		}
	})
	return max
}
