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

func (s Size) GetCount() int {
	count := 1
	for _, v := range s {
		count *= v
	}
	return count
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

func (d Data) ForEachIndex(f func(i, j, k int, value float64)) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				f(i, j, k, d.Value[i][j][k])
			}
		}
	}
}

func (d Data) MapIndex(f func(i, j, k int, value float64) float64) {
	for i := range d.Value {
		for j := range d.Value[i] {
			for k := range d.Value[i][j] {
				d.Value[i][j][k] = f(i, j, k, d.Value[i][j][k])
			}
		}
	}
}
