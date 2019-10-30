package core

import (
	"encoding/json"
	"fmt"
)

type (
	Describable interface {
		Desc() Desc
	}

	Desc interface {
		Brief() string
		Full() string
	}

	SimpleDesc struct {
		Name   string
		Core   string
		Params map[string]interface{}
	}
)

func (s SimpleDesc) Brief() string {
	if s.Core == "" {
		return s.Name
	}
	return fmt.Sprintf("%s (%s)", s.Name, s.Core)
}

func (s SimpleDesc) Full() string {
	if len(s.Params) == 0 {
		return s.Brief()
	}
	bytes, _ := json.MarshalIndent(s.Params, "", " ")
	return s.Brief() + string(bytes)
}
