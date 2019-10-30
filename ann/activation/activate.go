package activation

import (
	"github.com/XDean/go-machine-learning/ann/core"
)

type Activation interface {
	core.Describable
	Active(input float64) (output, partial float64)
}
