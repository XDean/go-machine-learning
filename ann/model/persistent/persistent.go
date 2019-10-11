package persistent

import (
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/model"
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

func Save(w *gob.Encoder, persistent Persistent) (err error) {
	defer model.RecoverNoError(&err)
	model.NoError(w.Encode(persistent.Name()))
	model.NoError(persistent.Save(w))
	return nil
}

func Load(w *gob.Decoder) (p Persistent, err error) {
	defer model.RecoverNoError(&err)
	name := ""
	model.NoError(w.Decode(&name))
	return New(name)
}

func TypeError(expect string, actual interface{}) error {
	return fmt.Errorf("Bad type: expect %s, but %T", expect, actual)
}
