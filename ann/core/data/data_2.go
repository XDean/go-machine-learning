package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
)

type Data2 struct {
	Size  []int
	Value [][]float64
}

func NewData2(x, y int) Data {
	result := Data2{
		Value: make([][]float64, x),
		Size:  []int{x, y},
	}
	for i := 0; i < x; i++ {
		result.Value[i] = make([]float64, y)
	}
	return result
}

func refData2(x, y int, value [][]float64) Data {
	return Data2{
		Size:  []int{x, y},
		Value: value,
	}
}

func (d Data2) SetValue(value float64, indexes []int) {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	d.Value[indexes[0]][indexes[1]] = value
}

func (d Data2) GetValue(indexes []int) float64 {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	return d.Value[indexes[0]][indexes[1]]
}

func (d Data2) GetData(indexes []int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		return refData1(d.Size[1], d.Value[indexes[0]])
	case 2:
		return refData0(&d.Value[indexes[0]][indexes[1]])
	default:
		panic("Can't get more than 2 dim data from Data2")
	}
}

func (d Data2) GetSize() []int {
	return d.Size
}

func (d Data2) GetCount() int {
	return d.Size[0] * d.Size[1]
}

func (d Data2) GetDim() int {
	return 2
}

func (d Data2) Fill(value float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			d.Value[i][j] = value
		}
	}
}

func (d Data2) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	for i, v := range d.Value {
		copy(result[i*d.Size[0]:], v)
	}
	return result
}

func (d Data2) ForEach(f func(value float64)) {
	for i := range d.Value {
		for j := range d.Value[i] {
			f(d.Value[i][j])
		}
	}
}

func (d Data2) Map(f func(value float64) float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			d.Value[i][j] = f(d.Value[i][j])
		}
	}
}

func (d Data2) ForEachIndex(f func(index []int, value float64)) {
	indexes := []int{0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			indexes[0], indexes[1] = i, j
			f(indexes, d.Value[i][j])
		}
	}
}

func (d Data2) MapIndex(f func(index []int, value float64) float64) {
	indexes := []int{0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			indexes[0], indexes[1] = i, j
			d.Value[i][j] = f(indexes, d.Value[i][j])
		}
	}
}
