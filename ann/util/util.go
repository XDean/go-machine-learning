package util

import "strings"

type noerr struct {
	error
}

func RecoverNoError(err *error) {
	r := recover()
	if e, ok := r.(noerr); ok {
		*err = e
	} else if r != nil {
		panic(r)
	}
}

func NoError(err error) {
	if err != nil {
		panic(noerr{err})
	}
}

func WriteWithPrefix(sb *strings.Builder, full string, prefix string) {
	split := strings.Split(full, "\n")
	for i, line := range split {
		if i != 0 {
			sb.WriteString(prefix)
		}
		sb.WriteString(line)
		if i != len(split)-1 {
			sb.WriteRune('\n')
		}
	}
}
