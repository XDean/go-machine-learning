package base

type noerr struct {
	error
}

func SizeToCount(size ...uint) uint {
	count := uint(1)
	for _, v := range size {
		count *= v
	}
	return count
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