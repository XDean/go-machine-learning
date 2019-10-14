package persistent

import (
	"encoding/gob"
)

func Register(o interface{}) {
	gob.Register(o)
}
