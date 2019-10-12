package base

type Data1 struct {
	len   int
	value []float64
}

func NewData1(len int) Data1 {
	return Data1{len: len, value: make([]float64, len)}
}

func (d Data1) SetValue(value float64, indexes ...int) Data {
	MustTrue(len(indexes) == 1)
	d.value[indexes[0]] = value
	return d
}

func (d Data1) GetValue(indexes ...int) float64 {
	MustTrue(len(indexes) == 1)
	return d.value[indexes[0]]
}

func (d Data1) GetData(indexes ...int) Data {
	switch len(indexes) {
	case 0:
		return d
	case 1:
		return NewData().SetValue(d.value[indexes[0]])
	default:
		panic("Can't get more than 1 dim data from Data2")
	}
}

func (d Data1) GetSize() []int {
	return []int{d.len}
}

func (d Data1) GetCount() int {
	return d.len
}

func (d Data1) GetDim() int {
	return 1
}

func (d Data1) Fill(value float64) Data {
	for i := range d.value {
		d.value[i] = value
	}
	return d
}

func (d Data1) ToArray() []float64 {
	result := make([]float64, d.len)
	copy(result, d.value)
	return result
}

func (d Data1) ForEach(f func(index []int, value float64)) {
	for i, v := range d.value {
		f([]int{i}, v)
	}
}
