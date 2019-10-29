package weight

type (
	Weight interface {
		Get() float64
		Set(value float64)
		Learn(gradient float64)
	}

	Factory interface {
		Create() Weight
	}
)
