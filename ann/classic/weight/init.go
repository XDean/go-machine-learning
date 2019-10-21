package weight

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
)

type Init interface {
	InitData(data Data)
	InitOne() float64
}
