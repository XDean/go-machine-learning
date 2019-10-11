package model

import "github.com/XDean/go-machine-learning/ann/model/persistent"

type ErrorFunc interface {
	persistent.Persistent
	Error(target, actual Data) (error float64, partial Data) // ∂E/∂a
}
