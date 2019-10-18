package data

import "github.com/XDean/go-machine-learning/ann/core/util"

type Sub struct {
	Actual Data
	Sub    []int
	Size   []int
}

func NewSub(actual Data, sub []int) Data {
	util.NoError(checkIndex(actual.GetSize(), sub, false))
	return Sub{Actual: actual, Sub: sub, Size: actual.GetSize()[len(sub):]}
}

func (d Sub) SetValue(value float64, indexes []int) Data {
	d.Actual.SetValue(value, append(d.Sub, indexes...))
	return d
}

func (d Sub) GetValue(indexes []int) float64 {
	return d.Actual.GetValue(append(d.Sub, indexes...))
}

func (d Sub) GetData(indexes []int) Data {
	return NewSub(
		d.Actual,
		append(d.Sub, indexes...),
	)
}

func (d Sub) GetSize() []int {
	return d.Size
}

func (d Sub) GetCount() int {
	size := d.GetSize()
	count := 1
	for _, v := range size {
		count *= v
	}
	return count
}

func (d Sub) GetDim() int {
	return d.Actual.GetDim() - len(d.Sub)
}

func (d Sub) Fill(value float64) Data {
	d.Map(func(_ float64) float64 { return value })
	return d
}

func (d Sub) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	i := 0
	d.ForEach(func(value float64) {
		result[i] = value
		i++
	})
	return result
}

func (d Sub) ForEach(f func(value float64)) {
	d.ForEachIndex(func(indexes []int, value float64) {
		f(value)
	})
}

func (d Sub) Map(f func(value float64) float64) {
	d.MapIndex(func(index []int, value float64) float64 {
		return f(value)
	})
}

func (d Sub) ForEachIndex(f func(indexes []int, value float64)) {
	forIndex(d.Size, func(index int, indexes []int) {
		f(append(d.Sub, indexes...), d.GetValue(indexes))
	})
}

func (d Sub) MapIndex(f func(index []int, value float64) float64) {
	forIndex(d.Size, func(index int, indexes []int) {
		d.SetValue(f(append(d.Sub, indexes...), d.GetValue(indexes)), indexes)
	})
}
