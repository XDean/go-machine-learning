package weight

import (
	"github.com/XDean/go-machine-learning/ann/persistent"
)

func init() {
	persistent.Register(&ZeroInit{})
}

type ZeroInit struct {
}

func (r *ZeroInit) Generate(count int) func() float64 {
	return func() float64 {
		return 0
	}
}
