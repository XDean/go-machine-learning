package weight

import (
	"github.com/XDean/go-machine-learning/ann/core"
)

type Init interface {
	InitData(data core.Data)
	InitOne() float64
}
