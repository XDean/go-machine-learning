package data

import (
	"fmt"
)

type DataRecursive struct {
	Len      int
	Children []Data
	Value    *float64
}

func NewDataRecusive(ls []int) Data {
	if len(ls) == 0 {
		value := 0.0
		return DataRecursive{Len: 0, Value: &value}
	}
	d := DataRecursive{
		Len:      ls[0],
		Children: make([]Data, ls[0]),
	}
	for i := range d.Children {
		d.Children[i] = NewDataRecusive(ls[1:])
	}
	return d
}

func (d DataRecursive) Fill(value float64) Data {
	d.ForEachIndex(func(index []int, _ float64) {
		d.SetValue(value, index)
	})
	return d
}

func (d DataRecursive) SetValue(value float64, indexes []int) Data {
	if len(indexes) == 0 {
		if d.isValue() {
			*d.Value = value
		} else {
			panic("This is not leaf, can't set")
		}
	} else {
		if indexes[0] < d.Len {
			d.Children[indexes[0]] = d.Children[indexes[0]].SetValue(value, indexes[1:])
		} else {
			panic(fmt.Sprintf("SetValue: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
	return d
}

func (d DataRecursive) GetValue(indexes []int) float64 {
	if len(indexes) == 0 {
		if d.isValue() {
			return *d.Value
		} else {
			panic("This is not leaf, use GetDataRecusive")
		}
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetValue(indexes[1:])
		} else {
			panic(fmt.Sprintf("GetValue: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d DataRecursive) GetData(indexes []int) Data {
	if len(indexes) == 0 {
		return d
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetData(indexes[1:])
		} else {
			panic(fmt.Sprintf("GetDataRecusive: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d DataRecursive) GetSize() []int {
	if d.Len == 0 {
		return []int{}
	} else {
		return append([]int{d.Len}, d.Children[0].GetSize()...)
	}
}

func (d DataRecursive) GetCount() int {
	if d.Len == 0 {
		return 1
	} else {
		return d.Len * d.Children[0].GetCount()
	}
}

func (d DataRecursive) GetDim() int {
	if d.Len == 0 {
		return 0
	} else {
		return 1 + d.Children[0].GetDim()
	}
}

func (d DataRecursive) ForEach(f func(value float64)) {
	if d.isValue() {
		f(*d.Value)
	} else {
		for _, v := range d.Children {
			v.ForEach(func(value float64) {
				f(value)
			})
		}
	}
}

func (d DataRecursive) Map(f func(value float64) float64) {
	if d.isValue() {
		*d.Value = f(*d.Value)
	} else {
		for _, v := range d.Children {
			v.Map(f)
		}
	}
}

func (d DataRecursive) ForEachIndex(f func(index []int, value float64)) {
	if d.isValue() {
		f([]int{}, *d.Value)
	} else {
		for i, v := range d.Children {
			v.ForEachIndex(func(index []int, value float64) {
				f(append([]int{int(i)}, index...), value)
			})
		}
	}
}

func (d DataRecursive) MapIndex(f func(index []int, value float64) float64) {
	if d.isValue() {
		*d.Value = f([]int{}, *d.Value)
	} else {
		for i, v := range d.Children {
			v.MapIndex(func(index []int, value float64) float64 {
				return f(append([]int{int(i)}, index...), value)
			})
		}
	}
}

func (d DataRecursive) ToArray() []float64 {
	if d.isValue() {
		return []float64{*d.Value}
	}
	count := d.GetCount()
	result := make([]float64, count)
	index := 0
	for _, v := range d.Children {
		array := v.ToArray()
		copy(result[index:], array)
		index += len(array)
	}
	return result
}

func (d DataRecursive) isValue() bool {
	return d.Len == 0
}
