package base

import (
	"github.com/XDean/go-machine-learning/ann/util"
	"reflect"
)

type DataReflect struct {
	size []int

	// for dim 0
	value *float64

	// for dim 1+
	ref    reflect.Value
	actual interface{}
}

func NewDataReflect(ls ...int) DataReflect {
	if ls == nil {
		ls = make([]int, 0)
	}
	result := DataReflect{size: ls}
	if len(ls) == 0 {
		value := 0.0
		result.value = &value
	} else {
		result.ref = makeSlice(ls)
		result.actual = result.ref.Interface()
	}
	return result
}

func (d DataReflect) Fill(value float64) Data {
	d.ForEach(func(index []int, _ float64) {
		d.SetValue(value, index...)
	})
	return d
}

func (d DataReflect) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.size, indexes, true))
	if d.isValue() {
		*d.value = value
	} else {
		d.findByIndex(indexes).SetFloat(value)
	}
	return d
}

func (d DataReflect) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.size, indexes, true))
	if d.isValue() {
		return *d.value
	} else {
		return d.findByIndex(indexes).Float()
	}
}

func (d DataReflect) GetData(indexes ...int) Data {
	if len(indexes) == 0 {
		return d
	}
	util.NoError(checkIndex(d.size, indexes, false))
	size := d.size[len(indexes):]
	result := NewDataReflect(size...)
	result.ForEach(func(index []int, _ float64) {
		result.SetValue(d.GetValue(append(indexes, index...)...), index...)
	})
	return result
}

func (d DataReflect) GetSize() []int {
	return d.size
}

func (d DataReflect) GetCount() int {
	count := int(1)
	for _, v := range d.size {
		count *= v
	}
	return count
}

func (d DataReflect) GetDim() int {
	return int(len(d.size))
}

func (d DataReflect) ForEach(f func(index []int, value float64)) {
	count := d.GetCount()
	for i := 0; i < count; i++ {
		index := indexToIndexes(d.size, i)
		f(index, d.GetValue(index...))
	}
}

func (d DataReflect) ToArray() []float64 {
	count := d.GetCount()
	result := make([]float64, count)
	for i := range result {
		indexes := indexToIndexes(d.size, count)
		result[i] = d.GetValue(indexes...)
	}
	return result
}

func (d DataReflect) Map(f func(index []int, value float64) float64) {
	if d.isValue() {
		*d.value = f([]int{}, *d.value)
	} else {
		count := d.GetCount()
		for i := 0; i < count; i++ {
			indexes := indexToIndexes(d.size, i)
			value := d.findByIndex(indexes)
			value.SetFloat(f(indexes, value.Float()))
		}
	}
}

func (d DataReflect) findByIndex(indexes []int) reflect.Value {
	result := d.ref
	for _, index := range indexes {
		result = result.Index(int(index))
	}
	return result
}

func (d DataReflect) isValue() bool {
	return len(d.size) == 0
}

func makeSlice(size []int) reflect.Value {
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
