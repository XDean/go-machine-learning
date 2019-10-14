package data

import (
	"github.com/XDean/go-machine-learning/ann/util"
)

type Data2 struct {
	X, Y  int
	Value [][]float64
}

func NewData2(x, y int) Data {
	result := Data2{
		X: x, Y: y,
		Value: make([][]float64, x),
	}
	for i := 0; i < x; i++ {
		result.Value[i] = make([]float64, y)
	}
	return result
}

func (d Data2) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	d.Value[indexes[0]][indexes[1]] = value
	return d
}

func (d Data2) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	return d.Value[indexes[0]][indexes[1]]
}

func (d Data2) GetData(indexes ...int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		result := NewData1(d.Y).(Data1)
		copy(result.Value, d.Value[indexes[0]])
		return result
	case 2:
		return NewData().SetValue(d.GetValue(indexes...))
	default:
		panic("Can't get more than 2 dim data from Data2")
	}
}

func (d Data2) GetSize() []int {
	return []int{d.X, d.Y}
}

func (d Data2) GetCount() int {
	return d.X * d.Y
}

func (d Data2) GetDim() int {
	return 2
}

func (d Data2) Fill(value float64) Data {
	for i := range d.Value {
		for j := range d.Value[i] {
			d.Value[i][j] = value
		}
	}
	return d
}

func (d Data2) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	for i, v := range d.Value {
		copy(result[i*d.X:], v)
	}
	return result
}

func (d Data2) ForEach(f func(index []int, value float64)) {
	indexes := []int{0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			indexes[0], indexes[1] = i, j
			f(indexes, d.Value[i][j])
		}
	}
}

func (d Data2) Map(f func(index []int, value float64) float64) {
	indexes := []int{0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			indexes[0], indexes[1] = i, j
			d.Value[i][j] = f(indexes, d.Value[i][j])
		}
	}
}
