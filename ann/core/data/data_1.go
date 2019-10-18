package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
)

type Data1 struct {
	Len   int
	Value []float64
}

func NewData1(len int) Data {
	return Data1{Len: len, Value: make([]float64, len)}
}

func refData1(len int, value []float64) Data {
	return Data1{Len: len, Value: value}
}

func (d Data1) SetValue(value float64, indexes []int) {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	d.Value[indexes[0]] = value
}

func (d Data1) GetValue(indexes []int) float64 {
	util.NoError(checkIndex(d.GetSize(), indexes, true))
	return d.Value[indexes[0]]
}

func (d Data1) GetData(indexes []int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		return refData0(&d.Value[indexes[0]])
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

func (d Data1) Fill(value float64) {
	for i := range d.Value {
		d.Value[i] = value
	}
}

func (d Data1) ToArray() []float64 {
	result := make([]float64, d.Len)
	copy(result, d.Value)
	return result
}

func (d Data1) ForEach(f func(value float64)) {
	for _, v := range d.Value {
		f(v)
	}
}

func (d Data1) Map(f func(value float64) float64) {
	for i, v := range d.Value {
		d.Value[i] = f(v)
	}
}

func (d Data1) ForEachIndex(f func(index []int, value float64)) {
	for i, v := range d.Value {
		f([]int{i}, v)
	}
}

func (d Data1) MapIndex(f func(index []int, value float64) float64) {
	for i, v := range d.Value {
		d.Value[i] = f([]int{i}, v)
	}
}
