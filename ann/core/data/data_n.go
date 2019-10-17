package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
)

type DataN struct {
	Size  []int
	Value []float64
}

func NewDataN(size ...int) Data {
	if size == nil {
		size = make([]int, 0)
	}
	count := 1
	for _, v := range size {
		count *= v
	}
	return DataN{
		Size:  size,
		Value: make([]float64, count),
	}
}

func (d DataN) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.Size, indexes, true))
	index := indexesToIndex(d.Size, indexes)
	d.Value[index] = value
	return d
}

func (d DataN) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.Size, indexes, true))
	index := indexesToIndex(d.Size, indexes)
	return d.Value[index]
}

func (d DataN) GetData(indexes ...int) Data {
	return NewSub(d, indexes)
}

func (d DataN) GetSize() []int {
	return d.Size
}

func (d DataN) GetCount() int {
	return len(d.Value)
}

func (d DataN) GetDim() int {
	return len(d.Size)
}

func (d DataN) Fill(value float64) Data {
	for i := range d.Value {
		d.Value[i] = value
	}
	return d
}

func (d DataN) ToArray() []float64 {
	result := make([]float64, len(d.Value))
	copy(result, d.Value)
	return result
}

func (d DataN) ForEach(f func(value float64)) {
	for _, v := range d.Value {
		f(v)
	}
}

func (d DataN) Map(f func(value float64) float64) {
	for i, v := range d.Value {
		d.Value[i] = f(v)
	}
}

func (d DataN) ForEachIndex(f func(indexes []int, value float64)) {
	forIndex(d.Size, func(index int, indexes []int) {
		f(indexes, d.Value[index])
	})
}

func (d DataN) MapIndex(f func(index []int, value float64) float64) {
	forIndex(d.Size, func(index int, indexes []int) {
		d.Value[index] = f(indexes, d.Value[index])
	})
}
