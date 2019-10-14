package persistent

import (
	"encoding/gob"
	"io"
)

type (
	Encoder interface {
		Encode(e interface{}) error
	}

	GobEncoder struct {
		*gob.Encoder
	}
)

func NewEncoder(w io.Writer) Encoder {
	return GobEncoder{gob.NewEncoder(w)}
}
