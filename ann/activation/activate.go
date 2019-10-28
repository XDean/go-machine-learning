package activation

type Activation interface {
	Active(input float64) (output, partial float64)
}
