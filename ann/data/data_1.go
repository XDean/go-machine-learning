package data

import (
	"github.com/XDean/go-machine-learning/ann/util"
)

type Data1 struct {
	Len   int
	Value []float64
}

func NewData1(len int) Data {
	return Data1{Len: len, Value: make([]float64, len)}
}

func (d Data1) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	d.Value[indexes[0]] = value
	return d
}

func (d Data1) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	return d.Value[indexes[0]]
}

func (d Data1) GetData(indexes ...int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		return NewData().SetValue(d.Value[indexes[0]])
	default:
		panic("Can't get more than 1 dim data from Data2")
	}
}

func (d Data1) GetSize() []int {
	return []int{d.Len}
}

func (d Data1) GetCount() int {
	return d.Len
}

func (d Data1) GetDim() int {
	return 1
}

func (d Data1) Fill(value float64) Data {
	for i := range d.Value {
		d.Value[i] = value
	}
	return d
}

func (d Data1) ToArray() []float64 {
	result := make([]float64, d.Len)
	copy(result, d.Value)
	return result
}

func (d Data1) ForEach(f func(index []int, value float64)) {
	for i, v := range d.Value {
		f([]int{i}, v)
	}
}

func (d Data1) Map(f func(index []int, value float64) float64) {
	for i, v := range d.Value {
		d.Value[i] = f([]int{i}, v)
	}
}
