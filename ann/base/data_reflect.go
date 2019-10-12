package base

import (
	"fmt"
	"reflect"
)

type DataReflect struct {
	size []uint

	// for dim 0
	value float64

	// for dim 1+
	ref    reflect.Value
	actual interface{}
}

func NewDataReflect(ls ...uint) DataReflect {
	if ls == nil {
		ls = make([]uint, 0)
	}
	result := DataReflect{size: ls}
	if len(ls) == 0 {
		result.value = 0.0
	} else {
		result.ref = makeSlice(ls)
		result.actual = result.ref.Interface()
	}
	return result
}

func makeSlice(size []uint) reflect.Value {
	t := reflect.TypeOf(float64(0.0))
	for i := 0; i < len(size); i++ {
		t = reflect.SliceOf(t)
	}
	result := reflect.MakeSlice(t, int(size[0]), int(size[0]))
	current := []reflect.Value{result}
	next := make([]reflect.Value, 0)
	for i := 1; i < len(size); i++ {
		for _, v := range current {
			elemType := v.Type().Elem()
			for j := 0; j < v.Len(); j++ {
				e := reflect.MakeSlice(elemType, int(size[i]), int(size[i]))
				v.Index(j).Set(e)
				next = append(next, e)
			}
		}
		current = next
		next = nil
	}
	return result
}

func (d DataReflect) IsLeaf() bool {
	return len(d.size) == 0
}

func (d DataReflect) Fill(value float64) DataReflect {
	d.ForEach(func(index []uint, _ float64) {
		d = d.SetValue(value, index...)
	})
	return d
}

func (d DataReflect) SetValue(value float64, indexes ...uint) DataReflect {
	NoError(d.checkIndex(indexes, true))
	if d.IsLeaf() {
		d.value = value
	} else {
		d.findByIndex(indexes).SetFloat(value)
	}
	return d
}

func (d DataReflect) GetValue(indexes ...uint) float64 {
	NoError(d.checkIndex(indexes, true))
	if d.IsLeaf() {
		return d.value
	} else {
		return d.findByIndex(indexes).Float()
	}
}

func (d DataReflect) GetDataReflect(indexes ...uint) DataReflect {
	NoError(d.checkIndex(indexes, false))
	size := d.size[len(indexes):]
	result := NewDataReflect(size...)
	result.ForEach(func(index []uint, _ float64) {
		result.SetValue(d.GetValue(append(indexes, index...)...), index...)
	})
	return result
}

func (d DataReflect) GetSize() []uint {
	return d.size
}

func (d DataReflect) GetCount() uint {
	count := uint(1)
	for _, v := range d.size {
		count *= v
	}
	return count
}

func (d DataReflect) GetDim() uint {
	return uint(len(d.size))
}

func (d DataReflect) ForEach(f func(index []uint, value float64)) {
	count := d.GetCount()
	for i := uint(0); i < count; i++ {
		index := d.indexToIndexes(i)
		f(index, d.GetValue(index...))
	}
}

func (d DataReflect) indexToIndexes(index uint) []uint {
	result := make([]uint, d.GetDim())
	for i := range d.size {
		size := d.size[len(d.size)-1-i]
		left := index % size
		index = (index - left) / size
		result[len(d.size)-1-i] = left
	}
	return result
}

func (d DataReflect) ToDim(dim uint) DataReflect {
	size := d.GetSize()
	if len(size) == int(dim) {
		return d
	}
	if dim == 0 {
		panic("Can't zip to dim 0")
	} else if dim == 1 {
		result := NewDataReflect(d.GetCount())
		i := uint(0)
		d.ForEach(func(_ []uint, value float64) {
			result.SetValue(value, i)
			i++
		})
		return result
	} else if d.IsLeaf() {
		size := make([]uint, dim)
		for i := range size {
			size[i] = 1
		}
		return NewDataReflect(size...).SetValue(d.GetValue(), make([]uint, dim)...)
	} else {
		count := d.GetDataReflect(make([]uint, dim-1)...).GetCount()
		result := NewDataReflect(append(append([]uint{}, d.size[:dim-1]...), count)...)
		pre := NewDataReflect(append(d.size[:dim-1])...)
		pre.ForEach(func(index []uint, value float64) {
			array := d.GetDataReflect(index...).ToArray()
			for i, v := range array {
				result = result.SetValue(v, append(index, uint(i))...)
			}
		})
		return result
	}
}

func (d DataReflect) Identity2D() DataReflect {
	result := NewDataReflect(append(d.GetSize(), d.GetSize()...)...)
	d.ForEach(func(index []uint, value float64) {
		result.SetValue(1, append(index, index...)...)
	})
	return result
}

func (d DataReflect) ToArray() []float64 {
	return d.ToDim(1).actual.([]float64)
}

func (d DataReflect) findByIndex(indexes []uint) reflect.Value {
	result := d.ref
	for _, index := range indexes {
		result = result.Index(int(index))
	}
	return result
}

func (d DataReflect) checkIndex(indexes []uint, match bool) error {
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
