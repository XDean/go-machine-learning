package base

import "fmt"

type DataRecusive struct {
	Len      uint
	Children []DataRecusive
	Value    float64
}

func NewDataRecusive(ls ...uint) DataRecusive {
	if len(ls) == 0 {
		return DataRecusive{Len: 0}
	}
	d := DataRecusive{
		Len:      ls[0],
		Children: make([]DataRecusive, ls[0]),
	}
	for i := range d.Children {
		d.Children[i] = NewDataRecusive(ls[1:]...)
	}
	return d
}

func (d DataRecusive) IsLeaf() bool {
	return d.Len == 0
}

func (d DataRecusive) Fill(value float64) DataRecusive {
	d.ForEach(func(index []uint, _ float64) {
		d = d.SetValue(value, index...)
	})
	return d
}

func (d DataRecusive) SetValue(value float64, indexes ...uint) DataRecusive {
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
			panic(fmt.Sprintf("SetValue: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
	return d
}

func (d DataRecusive) GetValue(indexes ...uint) float64 {
	if len(indexes) == 0 {
		if d.IsLeaf() {
			return d.Value
		} else {
			panic("This is not leaf, use GetDataRecusive")
		}
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetValue(indexes[1:]...)
		} else {
			panic(fmt.Sprintf("GetValue: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d DataRecusive) GetDataRecusive(indexes ...uint) DataRecusive {
	if len(indexes) == 0 {
		return d
	} else {
		if indexes[0] < d.Len {
			next := d.Children[indexes[0]]
			return next.GetDataRecusive(indexes[1:]...)
		} else {
			panic(fmt.Sprintf("GetDataRecusive: Index out of bound, len %d, get %d", d.Len, indexes[0]))
		}
	}
}

func (d DataRecusive) GetSize() []uint {
	if d.Len == 0 {
		return []uint{}
	} else {
		return append([]uint{d.Len}, d.Children[0].GetSize()...)
	}
}

func (d DataRecusive) GetCount() uint {
	if d.Len == 0 {
		return 1
	} else {
		return d.Len * d.Children[0].GetCount()
	}
}

func (d DataRecusive) GetDim() uint {
	if d.Len == 0 {
		return 0
	} else {
		return 1 + d.Children[0].GetDim()
	}
}

func (d DataRecusive) ForEach(f func(index []uint, value float64)) {
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

func (d DataRecusive) ToDim(dim uint) DataRecusive {
	size := d.GetSize()
	if len(size) == int(dim) {
		return d
	}
	if dim == 0 {
		panic("Can't zip to dim 0")
	} else if dim == 1 {
		result := NewDataRecusive(d.GetCount())
		i := 0
		d.ForEach(func(index []uint, value float64) {
			result.Children[i].Value = value
			i++
		})
		return result
	} else if d.IsLeaf() {
		result := NewDataRecusive(1)
		result.Children[0].Value = d.Value
		result.Children[0] = result.Children[0].ToDim(dim - 1)
		return result
	} else {
		result := NewDataRecusive(d.Len)
		for i, v := range d.Children {
			result.Children[i] = v.ToDim(dim - 1)
		}
		return result
	}
}

func (d DataRecusive) Identity2D() DataRecusive {
	result := NewDataRecusive(append(d.GetSize(), d.GetSize()...)...)
	d.ForEach(func(index []uint, value float64) {
		result.SetValue(1, append(index, index...)...)
	})
	return result
}

func (d DataRecusive) ToArray() []float64 {
	d1 := d.ToDim(1)
	result := make([]float64, d1.GetCount())
	d1.ForEach(func(index []uint, value float64) {
		result[index[0]] = value
	})
	return result
}
