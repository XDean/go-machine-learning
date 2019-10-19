package model

type (
	Size [3]int
	Data struct {
		Value [][][]float64 // chan * x * y
		Size  Size
	}
)

var EMPTY_DATA = NewData([3]int{0, 0, 0})

func NewData(size Size) Data {
	result := Data{
		Value: make([][][]float64, size[0]),
		Size:  size,
	}
	for i := range result.Value {
		result.Value[i] = make([][]float64, size[1])
		for j := range result.Value[i] {
			result.Value[i][j] = make([]float64, size[2])
		}
	}
	return result
}

func (d Data) GetValue(indexes []int) float64 {
	return d.Value[indexes[0]][indexes[1]][indexes[2]]
}

func (d Data) SetValue(value float64, indexes []int) {
	d.Value[indexes[0]][indexes[1]][indexes[2]] = value
}

func (d Data) ForEach(f func(value float64)) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				f(d.Value[i][j][k])
			}
		}
	}
}

func (d Data) Map(f func(value float64) float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				d.Value[i][j][k] = f(d.Value[i][j][k])
			}
		}
	}
}

func (d Data) ForEachIndex(f func(index []int, value float64)) {
	index := []int{0, 0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				index[0], index[1], index[2] = i, j, k
				f(index, d.Value[i][j][k])
			}
		}
	}
}

func (d Data) MapIndex(f func(index []int, value float64) float64) {
	index := []int{0, 0, 0}
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				index[0], index[1], index[2] = i, j, k
				d.Value[i][j][k] = f(index, d.Value[i][j][k])
			}
		}
	}
}
