package model

var constructors = make(map[string]func() Layer)

func RegisterLayer(name string, p func() Layer) {
	constructors[name] = p
}
