package base

import "errors"

func SizeToCount(size ...uint) uint {
	count := uint(1)
	for _, v := range size {
		count *= v
	}
	return count
}

func RecoverNoError(err *error) {
	r := recover()
	if e, ok := r.(error); ok {
		*err = e
	} else if e, ok := r.(string); ok {
		*err = errors.New(e)
	} else {
		panic(r)
	}
}

func NoError(err error) {
	if err != nil {
		panic(err)
	}
}
