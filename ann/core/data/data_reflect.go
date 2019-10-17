package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
	"reflect"
)

// gob can't handle this
type DataReflect struct {
	Size []int

	// for dim 0
	Value *float64

	// for dim 1+
	Ref    reflect.Value
	Actual interface{}
}

func NewDataReflect(ls ...int) DataReflect {
	if ls == nil {
		ls = make([]int, 0)
	}
	result := DataReflect{Size: ls}
	if len(ls) == 0 {
		value := 0.0
		result.Value = &value
	} else {
		result.Ref = makeSlice(ls)
		result.Actual = result.Ref.Interface()
	}
	return result
}

func (d DataReflect) Fill(value float64) Data {
	d.ForEachIndex(func(index []int, _ float64) {
		d.SetValue(value, index...)
	})
	return d
}

func (d DataReflect) SetValue(value float64, indexes ...int) Data {
	util.NoError(checkIndex(d.Size, indexes, true))
	if d.isValue() {
		*d.Value = value
	} else {
		d.findByIndex(indexes).SetFloat(value)
	}
	return d
}

func (d DataReflect) GetValue(indexes ...int) float64 {
	util.NoError(checkIndex(d.Size, indexes, true))
	if d.isValue() {
		return *d.Value
	} else {
		return d.findByIndex(indexes).Float()
	}
}

func (d DataReflect) GetData(indexes ...int) Data {
	return NewSub(d, indexes)
}

func (d DataReflect) GetSize() []int {
	return d.Size
}

func (d DataReflect) GetCount() int {
	count := int(1)
	for _, v := range d.Size {
		count *= v
	}
	return count
}

func (d DataReflect) GetDim() int {
	return int(len(d.Size))
}

func (d DataReflect) ToArray() []float64 {
	count := d.GetCount()
	result := make([]float64, count)
	for i := range result {
		indexes := indexToIndexes(d.Size, count)
		result[i] = d.GetValue(indexes...)
	}
	return result
}

func (d DataReflect) ForEach(f func(value float64)) {
	count := d.GetCount()
	for i := 0; i < count; i++ {
		index := indexToIndexes(d.Size, i)
		f(d.GetValue(index...))
	}
}

func (d DataReflect) Map(f func(value float64) float64) {
	if d.isValue() {
		*d.Value = f(*d.Value)
	} else {
		count := d.GetCount()
		for i := 0; i < count; i++ {
			indexes := indexToIndexes(d.Size, i)
			value := d.findByIndex(indexes)
			value.SetFloat(f(value.Float()))
		}
	}
}

func (d DataReflect) ForEachIndex(f func(index []int, value float64)) {
	count := d.GetCount()
	for i := 0; i < count; i++ {
		index := indexToIndexes(d.Size, i)
		f(index, d.GetValue(index...))
	}
}

func (d DataReflect) MapIndex(f func(index []int, value float64) float64) {
	if d.isValue() {
		*d.Value = f([]int{}, *d.Value)
	} else {
		count := d.GetCount()
		for i := 0; i < count; i++ {
			indexes := indexToIndexes(d.Size, i)
			value := d.findByIndex(indexes)
			value.SetFloat(f(indexes, value.Float()))
		}
	}
}

func (d DataReflect) findByIndex(indexes []int) reflect.Value {
	result := d.Ref
	for _, index := range indexes {
		result = result.Index(int(index))
	}
	return result
}

func (d DataReflect) isValue() bool {
	return len(d.Size) == 0
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
