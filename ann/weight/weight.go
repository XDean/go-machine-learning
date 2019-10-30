package weight

import "github.com/XDean/go-machine-learning/ann/core"

type (
	Weight interface {
		Get() float64
		Set(value float64)
		Learn(gradient float64)
	}

	Factory interface {
		core.Describable
		Create() Weight
	}
)
