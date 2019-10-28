package util

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
