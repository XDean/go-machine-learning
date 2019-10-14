package model

import (
	"github.com/XDean/go-machine-learning/ann/data"
)

type ErrorFunc interface {
	CalcError(target, actual data.Data) (error float64, partial data.Data) // ∂E/∂a
}
