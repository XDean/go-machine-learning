package persistent

import "encoding/gob"

type TypePersistent struct{}

func (TypePersistent) Save(writer *gob.Encoder) error {
	return nil
}

func (TypePersistent) Load(reader *gob.Decoder) error {
	return nil
}
