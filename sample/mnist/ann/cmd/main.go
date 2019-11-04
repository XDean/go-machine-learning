package main

import (
	"github.com/XDean/go-machine-learning/sample/mnist/ann"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().Unix())
	registerModels()
	mnist_ann.Main()
}
