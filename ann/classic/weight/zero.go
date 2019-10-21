package weight

import (
	. "github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

func init() {
	persistent.Register(&ZeroInit{})
}

type ZeroInit struct {
}

func (r *ZeroInit) InitOne() float64 {
	return 0
}

func (r *ZeroInit) InitData(data Data) {
	data.Map(func(_ float64) float64 {
		return 0
	})
}
