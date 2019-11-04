package main

import (
	"github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/sample/mnist"
)

func mnistToData(d mnist.Data) (input, target core.Data) {
	input = core.NewData([3]int{1, 28, 28})
	target = core.NewData([3]int{1, 1, 10})

	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			input.Value[0][i][j] = float64(d.Image[i*28+j]) / 255.0
		}
	}
	target.Value[0][0][d.Label] = 1
	return
}

func expectFromResult(r core.Result) int {
	max := 0
	for i, v := range r.Target.Value[0][0] {
		if v > r.Target.Value[0][0][max] {
			max = i
		}
	}
	return max
}

func predictFromResult(r core.Result) int {
	max := 0
	for i, v := range r.Output.Value[0][0] {
		if v > r.Output.Value[0][0][max] {
			max = i
		}
	}
	return max
}

func adapt(mnistStream <-chan mnist.Data) <-chan core.TrainData {
	result := make(chan core.TrainData, 10)
	go func() {
		for {
			data, ok := <-mnistStream
			if !ok {
				break
			}
			input, target := mnistToData(data)
			result <- core.TrainData{Input: input, Target: target}
		}
		close(result)
	}()
	return result
}
