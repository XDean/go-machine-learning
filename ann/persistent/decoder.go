package persistent

import (
	"encoding/gob"
	"io"
)

type (
	Decoder interface {
		Decode(e interface{}) error
	}

	GobDecoder struct {
		*gob.Decoder
	}
)

func NewDecoder(r io.Reader) GobDecoder {
	return GobDecoder{gob.NewDecoder(r)}
}
