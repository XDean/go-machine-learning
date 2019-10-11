package model

import (
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/model/persistent"
)

type ErrorFunc interface {
	persistent.Persistent
	CalcError(target, actual base.Data) (error float64, partial base.Data) // ∂E/∂a
}
