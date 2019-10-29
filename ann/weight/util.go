package weight

import "github.com/XDean/go-machine-learning/ann/core"

func Create(factory Factory, init Init) Weight {
	result := factory.Create()
	result.Set(init.Generate(1)())
	return result
}

func Create1D(factory Factory, init Init, length int) []Weight {
	result := make([]Weight, length)
	initFunc := init.Generate(length)
	for i := range result {
		w := factory.Create()
		w.Set(initFunc())
		result[i] = w
	}
	return result
}

func Create2D(factory Factory, init Init, x, y int) [][]Weight {
	result := make([][]Weight, x)
	count := x * y
	initFunc := init.Generate(count)
	for i := range result {
		result[i] = make([]Weight, y)
		for j := range result[i] {
			w := factory.Create()
			w.Set(initFunc())
			result[i][j] = w
		}
	}
	return result
}

func Create3D(factory Factory, init Init, size core.Size) [][][]Weight {
	result := make([][][]Weight, size[0])
	count := size.GetCount()
	initFunc := init.Generate(count)
	for i := range result {
		result[i] = make([][]Weight, size[1])
		for j := range result[i] {
			result[i][j] = make([]Weight, size[2])
			for k := range result[i][j] {
				w := factory.Create()
				w.Set(initFunc())
				result[i][j][k] = w
			}
		}
	}
	return result
}
