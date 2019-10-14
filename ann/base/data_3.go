package base

import "github.com/XDean/go-machine-learning/ann/util"

type Data3 struct {
	x, y, z int
	value   [][][]float64
}

func NewData3(x, y, z int) Data {
	result := Data3{
		x: x, y: y, z: z,
		value: make([][][]float64, x),
	}
	for i := 0; i < x; i++ {
		result.value[i] = make([][]float64, y)
		for j := 0; j < y; j++ {
			result.value[i][j] = make([]float64, z)
		}
	}
	return result
}

func (d Data3) SetValue(value float64, indexes ...int) Data {
	util.MustTrue(len(indexes) == 3)
	d.value[indexes[0]][indexes[1]][indexes[2]] = value
	return d
}

func (d Data3) GetValue(indexes ...int) float64 {
	util.MustTrue(len(indexes) == 3)
	return d.value[indexes[0]][indexes[1]][indexes[2]]
}

func (d Data3) GetData(indexes ...int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		result := NewData2(d.y, d.z).(Data2)
		copy(result.value, d.value[indexes[0]])
		return result
	case 2:
		result := NewData1(d.z).(Data1)
		copy(result.value, d.value[indexes[0]][indexes[1]])
		return result
	case 3:
		return NewData().SetValue(d.GetValue(indexes...))
	default:
		panic("Can't get more than 3 dim data from Data3")
	}
}

func (d Data3) GetSize() []int {
	return []int{d.x, d.y, d.z}
}

func (d Data3) GetCount() int {
	return d.x * d.y * d.z
}

func (d Data3) GetDim() int {
	return 3
}

func (d Data3) Fill(value float64) Data {
	for i := range d.value {
		for j := range d.value[i] {
			for k := range d.value[i][j] {
				d.value[i][j][k] = value
			}
		}
	}
	return d
}

func (d Data3) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	for i := range d.value {
		for j := range d.value[i] {
			copy(result[indexesToIndex(d.GetSize(), []int{i, j}):], d.value[i][j])
		}
	}
	return result
}

func (d Data3) ForEach(f func(index []int, value float64)) {
	for i := range d.value {
		for j := range d.value[i] {
			for k := range d.value[i][j] {
				f([]int{i, j, k}, d.value[i][j][k])
			}
		}
	}
}

func (d Data3) Map(f func(index []int, value float64) float64) {
	for i := range d.value {
		for j := range d.value[i] {
			for k := range d.value[i][j] {
				d.value[i][j][k] = f([]int{i, j}, d.value[i][j][k])
			}
		}
	}
}
