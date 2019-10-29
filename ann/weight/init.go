package weight

type Init interface {
	Generate(count int) func() float64
}
