package persistent

import (
	"encoding/gob"
	"errors"
	"fmt"
)

type Persistent interface {
	Name() string

	Save(writer *gob.Encoder) error
	Load(reader *gob.Decoder) error
}

var constructors = make(map[string]func() Persistent)

func Register(constructor func() Persistent) {
	name := constructor().Name()
	if constructors[name] != nil {
		fmt.Println("Duplicate register: " + name)
	}
	constructors[name] = constructor
}

func New(name string) (Persistent, error) {
	if con, ok := constructors[name]; ok {
		return con(), nil
	} else {
		return nil, errors.New("No such persistent bean: " + name)
	}
}
