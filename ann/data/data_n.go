package data

import (
	"github.com/XDean/go-machine-learning/ann/util"
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
	size := d.Size[len(indexes):]
	result := NewDataN(size...).(DataN)
	startIndexes := make([]int, d.GetDim())
	copy(startIndexes[:len(indexes)], indexes)
	startIndex := indexesToIndex(d.Size, startIndexes)
	copy(result.Value, d.Value[startIndex:startIndex+len(result.Value)])
	return result
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

func (d DataN) ForEach(f func(indexes []int, value float64)) {
	forIndex(d.Size, func(i indexPair) {
		f(i.indexes, d.Value[i.index])
	})
}

func (d DataN) Map(f func(index []int, value float64) float64) {
	forIndex(d.Size, func(i indexPair) {
		d.Value[i.index] = f(i.indexes, d.Value[i.index])
	})
}
