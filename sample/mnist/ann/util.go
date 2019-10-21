package main

import (
	"github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/sample/mnist"
)

func mnistToData(d mnist.Data) (input, target model.Data) {
	input = model.NewData([3]int{1, 28, 28})
	target = model.NewData([3]int{1, 1, 10})

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			input.Value[0][i][j] = float64(d.Image[i*28+j]) / 255.0
		}
	}
	target.Value[0][0][d.Label] = 1
	return
}

func predictFromResult(r model.Result) int {
	max := 0
	for i, v := range r.Output.Value[0][0] {
		if v > r.Output.Value[0][0][max] {
			max = i
		}
	}
	return max
}
