package base

import (
	"github.com/XDean/go-machine-learning/ann/util"
)

type DataN struct {
	size  []int
	value []float64
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
		size:  size,
		value: make([]float64, count),
	}
}

func (d DataN) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.size, indexes, true))
	index := indexesToIndex(d.size, indexes)
	d.value[index] = value
	return d
}

func (d DataN) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.size, indexes, true))
	index := indexesToIndex(d.size, indexes)
	return d.value[index]
}

func (d DataN) GetData(indexes ...int) Data {
	size := d.size[len(indexes):]
	result := NewDataN(size...).(DataN)
	startIndexes := make([]int, d.GetDim())
	copy(startIndexes[:len(indexes)], indexes)
	startIndex := indexesToIndex(d.size, startIndexes)
	copy(result.value, d.value[startIndex:startIndex+len(result.value)])
	return result
}

func (d DataN) GetSize() []int {
	return d.size
}

func (d DataN) GetCount() int {
	return len(d.value)
}

func (d DataN) GetDim() int {
	return len(d.size)
}

func (d DataN) Fill(value float64) Data {
	for i := range d.value {
		d.value[i] = value
	}
	return d
}

func (d DataN) ToArray() []float64 {
	result := make([]float64, len(d.value))
	copy(result, d.value)
	return result
}

func (d DataN) ForEach(f func(indexes []int, value float64)) {
	forIndex(d.size, func(i indexPair) {
		f(i.indexes, d.value[i.index])
	})
}

func (d DataN) Map(f func(index []int, value float64) float64) {
	forIndex(d.size, func(i indexPair) {
		d.value[i.index] = f(i.indexes, d.value[i.index])
	})
}
