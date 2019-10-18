package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
)

type Data3 struct {
	X, Y, Z int
	Value   [][][]float64
}

func NewData3(x, y, z int) Data {
	result := Data3{
		X: x, Y: y, Z: z,
		Value: make([][][]float64, x),
	}
	for i := 0; i < x; i++ {
		result.Value[i] = make([][]float64, y)
		for j := 0; j < y; j++ {
			result.Value[i][j] = make([]float64, z)
		}
	}
	return result
}

func (d Data3) SetValue(value float64, indexes []int) {
	util.MustTrue(len(indexes) == 3)
	d.Value[indexes[0]][indexes[1]][indexes[2]] = value
}

func (d Data3) GetValue(indexes []int) float64 {
	util.MustTrue(len(indexes) == 3)
	return d.Value[indexes[0]][indexes[1]][indexes[2]]
}

func (d Data3) GetData(indexes []int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		result := NewData2(d.Y, d.Z).(Data2)
		copy(result.Value, d.Value[indexes[0]])
		return result
	case 2:
		result := NewData1(d.Z).(Data1)
		copy(result.Value, d.Value[indexes[0]][indexes[1]])
		return result
	case 3:
		result := NewData0()
		result.SetValue(d.GetValue(indexes), nil)
		return result
	default:
		panic("Can't get more than 3 dim data from Data3")
	}
}

func (d Data3) GetSize() []int {
	return []int{d.X, d.Y, d.Z}
}

func (d Data3) GetCount() int {
	return d.X * d.Y * d.Z
}

func (d Data3) GetDim() int {
	return 3
}

func (d Data3) Fill(value float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				d.Value[i][j][k] = value
			}
		}
	}
}

func (d Data3) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	for i := range d.Value {
		for j := range d.Value[i] {
			copy(result[indexesToIndex(d.GetSize(), []int{i, j}):], d.Value[i][j])
		}
	}
	return result
}

func (d Data3) ForEach(f func(value float64)) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				f(d.Value[i][j][k])
			}
		}
	}
}

func (d Data3) Map(f func(value float64) float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				d.Value[i][j][k] = f(d.Value[i][j][k])
			}
		}
	}
}

func (d Data3) ForEachIndex(f func(index []int, value float64)) {
	index := []int{0, 0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				index[0], index[1], index[2] = i, j, k
				f(index, d.Value[i][j][k])
			}
		}
	}
}

func (d Data3) MapIndex(f func(index []int, value float64) float64) {
	index := []int{0, 0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				index[0], index[1], index[2] = i, j, k
				d.Value[i][j][k] = f(index, d.Value[i][j][k])
			}
		}
	}
}
