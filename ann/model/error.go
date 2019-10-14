package model

import (
	"github.com/XDean/go-machine-learning/ann/base"
)

type ErrorFunc interface {
	CalcError(target, actual base.Data) (error float64, partial base.Data) // ∂E/∂a
}
