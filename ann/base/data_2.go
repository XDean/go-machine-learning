package base

type Data2 struct {
	x, y  int
	value [][]float64
}

func NewData2(x, y int) Data2 {
	result := Data2{
		x: x, y: y,
		value: make([][]float64, x),
	}
	for i := 0; i < x; i++ {
		result.value[i] = make([]float64, y)
	}
	return result
}

func (d Data2) SetValue(value float64, indexes ...int) Data {
	MustTrue(len(indexes) == 2)
	d.value[indexes[0]][indexes[1]] = value
	return d
}

func (d Data2) GetValue(indexes ...int) float64 {
	MustTrue(len(indexes) == 2)
	return d.value[indexes[0]][indexes[1]]
}

func (d Data2) GetData(indexes ...int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		result := NewData1(d.y)
		copy(result.value, d.value[indexes[0]])
		return result
	case 2:
		return NewData().SetValue(d.GetValue(indexes...))
	default:
		panic("Can't get more than 2 dim data from Data2")
	}
}

func (d Data2) GetSize() []int {
	return []int{d.x, d.y}
}

func (d Data2) GetCount() int {
	return d.x * d.y
}

func (d Data2) GetDim() int {
	return 2
}

func (d Data2) Fill(value float64) Data {
	for i := range d.value {
		for j := range d.value[i] {
			d.value[i][j] = value
		}
	}
	return d
}

func (d Data2) ToArray() []float64 {
	result := make([]float64, d.GetCount())
	for i, v := range d.value {
		copy(result[i*d.x:], v)
	}
	return result
}

func (d Data2) ForEach(f func(index []int, value float64)) {
	for i := range d.value {
		for j := range d.value[i] {
			f([]int{i, j}, d.value[i][j])
		}
	}
}
