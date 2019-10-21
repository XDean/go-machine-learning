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

func MustTrue(b bool, msg ...string) {
	if !b {
		if len(msg) > 0 {
			panic(msg[0])
		} else {
			panic("Never happen")
		}
	}
}
