package core

import (
	"fmt"
	"sort"
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
		} else if d, ok := s.Core.(fmt.Stringer); ok {
			core = d.String()
		}
		return fmt.Sprintf("%s (%v)", s.Name, core)
	}
}

func (s SimpleDesc) Full() string {
	sb := strings.Builder{}

	sb.WriteString(s.Name)

	if s.Core != nil {
		core := s.Core
		if d, ok := s.Core.(Describable); ok {
			core = d.Desc().Brief()
		} else if d, ok := s.Core.(fmt.Stringer); ok {
			core = d.String()
		}
		sb.WriteString(fmt.Sprintf("(%v)", core))
	}
	paramLen := len(s.Params)
	if paramLen != 0 {
		i := 0

		keys := make([]string, paramLen)
		for k := range s.Params {
			keys[i] = k
			i++
		}
		sort.Strings(keys)

		sb.WriteString(" [")
		for index, k := range keys {
			v := s.Params[k]
			sb.WriteString(k)
			sb.WriteRune('=')
			if d, ok := v.(Describable); ok {
				sb.WriteRune('{')
				sb.WriteString(d.Desc().Full())
				sb.WriteRune('}')
			} else if d, ok := s.Core.(fmt.Stringer); ok {
				sb.WriteString(d.String())
			} else {
				sb.WriteString(fmt.Sprintf("%v", v))
			}
			if index != paramLen-1 {
				sb.WriteString(", ")
			}
		}
		sb.WriteString("]")
	}
	return sb.String()
}
