package model

import "fmt"

type Data struct {
	Len      uint
	Children []Data
	Value    float64
}

func NewData(ls ...uint) Data {
	if len(ls) == 0 {
		return Data{Len: 0}
	}
	d := Data{
		Len:      ls[0],
		Children: make([]Data, ls[0]),
	}
	for i := range d.Children {
		d.Children[i] = NewData(ls[1:]...)
	}
	return d
}

func (d Data) IsLeaf() bool {
	return d.Len == 0
}

func (d Data) SetValue(value float64, indexes ...uint) Data {
	if len(indexes) == 0 {
		if d.IsLeaf() {
			d.Value = value
		} else {
			panic("This is not leaf, can't set")
		}
	} else {
		if indexes[0] < d.Len {
			d.Children[indexes[0]] = d.Children[indexes[0]].SetValue(value, indexes[1:]...)
		} else {
			panic(fmt.Sprintf("Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
	return d
}

func (d Data) GetValue(indexes ...uint) float64 {
	if len(indexes) == 0 {
		if d.IsLeaf() {
			return d.Value
		} else {
			panic("This is not leaf, use GetData")
		}
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetValue(indexes[1:]...)
		} else {
			panic(fmt.Sprintf("Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d Data) GetData(indexes ...uint) Data {
	if len(indexes) == 0 {
		return d
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetData(indexes[1:]...)
		} else {
			panic(fmt.Sprintf("Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d Data) GetSize() []uint {
	if d.Len == 0 {
		return []uint{}
	} else {
		return append([]uint{d.Len}, d.Children[0].GetSize()...)
	}
}

func (d Data) GetCount() uint {
	if d.Len == 0 {
		return 1
	} else {
		return d.Len * d.Children[0].GetCount()
	}
}

func (d Data) GetDim() uint {
	if d.Len == 0 {
		return 0
	} else {
		return 1 + d.Children[0].GetDim()
	}
}

func (d Data) ForEach(f func(index []uint, value float64)) {
	if d.IsLeaf() {
		f(nil, d.Value)
	} else {
		for i, v := range d.Children {
			v.ForEach(func(index []uint, value float64) {
				f(append([]uint{uint(i)}, index...), value)
			})
		}
	}
}

func (d Data) ToDim(dim int) Data {
	size := d.GetSize()
	if len(size) == dim {
		return d
	}
	if dim == 1 {
		result := NewData(d.GetCount())
		i := 0
		d.ForEach(func(index []uint, value float64) {
			result.Children[i].Value = value
			i++
		})
		return result
	} else if d.IsLeaf() {
		result := NewData(1)
		result.Children[0].Value = d.Value
		result.Children[0] = result.Children[0].ToDim(dim - 1)
		return result
	} else {
		result := NewData(d.Len)
		for i, v := range d.Children {
			result.Children[i] = v.ToDim(dim - 1)
		}
		return result
	}
}
