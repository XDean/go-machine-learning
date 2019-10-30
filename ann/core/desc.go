package core

import (
	"fmt"
	"strings"
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
		Core   interface{}
		Params map[string]interface{}
	}
)

func (s SimpleDesc) Brief() string {
	if s.Core == nil {
		return s.Name
	} else {
		core := s.Core
		if d, ok := s.Core.(Describable); ok {
			core = d.Desc().Brief()
		}
		return fmt.Sprintf("%s (%s)", s.Name, core)
	}
}

func (s SimpleDesc) Full() string {
	sb := strings.Builder{}

	sb.WriteString(s.Name)

	if s.Core != nil {
		core := s.Core
		if d, ok := s.Core.(Describable); ok {
			core = d.Desc().Brief()
		}
		sb.WriteString(fmt.Sprintf("(%s)", core))
	}
	if len(s.Params) != 0 {
		for k, v := range s.Params {
			sb.WriteString(k)
			sb.WriteRune('=')
			if d, ok := v.(Describable); ok {
				sb.WriteRune('{')
				sb.WriteString(d.Desc().Full())
				sb.WriteRune('}')
			} else {
				sb.WriteString(fmt.Sprintf("%s", v))
			}
			sb.WriteString(", ")
		}
	}
	return sb.String()
}
