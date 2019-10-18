package data

import (
	"github.com/XDean/go-machine-learning/ann/core/persistent"
)

type Data interface {
	SetValue(value float64, indexes []int) Data // return self
	GetValue(indexes []int) float64
	GetData(indexes []int) Data

	GetSize() []int
	GetCount() int
	GetDim() int

	Fill(value float64) Data // return self
	ToArray() []float64

	ForEachIndex(f func(indexes []int, value float64))
	MapIndex(f func(indexes []int, value float64) float64)
	ForEach(f func(value float64))
	Map(f func(value float64) float64)
}

func init() {
	persistent.Register(Data0{})
	persistent.Register(Data1{})
	persistent.Register(Data2{})
	persistent.Register(Data3{})
	persistent.Register(DataN{})
	persistent.Register(DataRecursive{})
}

func NewData(size []int) Data {
	switch len(size) {
	case 0:
		return NewData0()
	case 1:
		return NewData1(size[0])
	case 2:
		return NewData2(size[0], size[1])
	case 3:
		return NewData3(size[0], size[1], size[2])
	default:
		return NewDataN(size)
	}
}

func Identity2D(d Data) Data {
	result := NewData(append(d.GetSize(), d.GetSize()...))
	d.ForEachIndex(func(index []int, value float64) {
		result.SetValue(1, append(index, index...))
	})
	return result
}

func ToDim(d Data, dim int) Data {
	size := d.GetSize()
	if len(size) == int(dim) {
		return d
	}
	if dim == 0 {
		panic("Can't zip to dim 0")
	} else if dim == 1 {
		result := NewData([]int{d.GetCount()})
		array := d.ToArray()
		for i, v := range array {
			result.SetValue(v, []int{i})
		}
		return result
	} else if d.GetDim() == 0 {
		size := make([]int, dim)
		for i := range size {
			size[i] = 1
		}
		result := NewData(size)
		result.SetValue(d.GetValue(nil), make([]int, dim))
		return result
	} else {
		count := d.GetData(make([]int, dim-1)).GetCount()
		result := NewData(append(append([]int{}, size[:dim-1]...), count))
		pre := NewData(append(size[:dim-1]))
		pre.ForEachIndex(func(index []int, value float64) {
			array := d.GetData(index).ToArray()
			for i, v := range array {
				result.SetValue(v, append(index, int(i)))
			}
		})
		return result
	}
}
