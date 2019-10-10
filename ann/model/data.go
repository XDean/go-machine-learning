package model

import "fmt"

type Data struct {
	Len   uint
	Next  []*Data
	Value float64
}

func NewData(l uint, ls ...uint) *Data {
	d := &Data{
		Len:  l,
		Next: make([]*Data, l),
	}
	if len(ls) == 0 {
		for i := range d.Next {
			d.Next[i] = &Data{Len: 0}
		}
	} else {
		for i := range d.Next {
			d.Next[i] = NewData(ls[0], ls[1:]...)
		}
	}
	return d
}

func (d *Data) IsLeaf() bool {
	return d.Len == 0
}

func (d *Data) GetValue(index uint, indexes ...uint) float64 {
	if len(indexes) == 0 {
		if d.IsLeaf() {
			return d.Value
		} else {
			panic("Data is not leaf, use GetData")
		}
	} else {
		if index < d.Len {
			next := d.Next[index]
			return next.GetValue(indexes[0], indexes[1:]...)
		} else {
			panic(fmt.Sprintf("Index out of bound, len %d, get %d", d.Len, index))
		}
	}
}

func (d *Data) GetData(index uint, indexes ...uint) *Data {
	if len(indexes) == 0 {
		if d.IsLeaf() {
			panic("Data is leaf, use GetValue")
		} else {
			return d.Next[index]
		}
	} else {
		if index < d.Len {
			next := d.Next[index]
			return next.GetData(indexes[0], indexes[1:]...)
		} else {
			panic(fmt.Sprintf("Index out of bound, len %d, get %d", d.Len, index))
		}
	}
}

func (d *Data) GetDimension() []uint {
	if d.Len == 0 {
		return nil
	} else {
		return append([]uint{d.Len}, d.Next[0].GetDimension()...)
	}
}

func (d *Data) ForEach(f func(index []uint, value *float64)) {
	if d.IsLeaf() {
		f(nil, &d.Value)
	} else {
		for i, v := range d.Next {
			v.ForEach(func(index []uint, value *float64) {
				f(append([]uint{uint(i)}, index...), value)
			})
		}
	}
}
