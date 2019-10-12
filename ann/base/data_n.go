package base

import "fmt"

type DataN struct {
	size  []int
	value []float64
}

func NewDataN(size ...int) DataN {
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
	NoError(d.checkIndex(indexes, true))
	index := d.indexesToIndex(indexes)
	d.value[index] = value
	return d
}

func (d DataN) GetValue(indexes ...int) float64 {
	NoError(d.checkIndex(indexes, true))
	index := d.indexesToIndex(indexes)
	return d.value[index]
}

func (d DataN) GetData(indexes ...int) Data {
	size := d.size[len(indexes):]
	result := NewDataN(size...)
	startIndexes := make([]int, d.GetDim())
	copy(startIndexes[:len(indexes)], indexes)
	startIndex := d.indexesToIndex(startIndexes)
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
	for i, v := range d.value {
		f(d.indexToIndexes(i), v)
	}
}

func (d DataN) checkIndex(indexes []int, match bool) error {
	if match && len(indexes) != len(d.size) {
		return fmt.Errorf("Index not match, actual %d, get %d", len(d.size), len(indexes))
	}
	for i, v := range indexes {
		if v >= d.size[i] {
			return fmt.Errorf("Index out of bound: actual %v, get %v", d.size, indexes)
		}
	}
	return nil
}

func (d DataN) indexesToIndex(indexes []int) int {
	index := 0
	for i, v := range indexes {
		index = d.size[i]*index + v
	}
	return index
}

func (d DataN) indexToIndexes(index int) []int {
	result := make([]int, d.GetDim())
	for i := range d.size {
		size := d.size[len(d.size)-1-i]
		left := index % size
		index = (index - left) / size
		result[len(d.size)-1-i] = left
	}
	return result
}
