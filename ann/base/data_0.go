package base

type Data0 struct {
	value *float64
}

func NewData0() Data0 {
	value := 0.0
	return Data0{value: &value}
}

func (d Data0) SetValue(value float64, indexes ...int) Data {
	MustTrue(len(indexes) == 0)
	*d.value = value
	return d
}

func (d Data0) GetValue(indexes ...int) float64 {
	MustTrue(len(indexes) == 0)
	return *d.value
}

func (d Data0) GetData(indexes ...int) Data {
	MustTrue(len(indexes) == 0)
	return d
}

func (d Data0) GetSize() []int {
	return make([]int, 0)
}

func (d Data0) GetCount() int {
	return 1
}

func (d Data0) GetDim() int {
	return 0
}

func (d Data0) Fill(value float64) Data {
	*d.value = value
	return d
}

func (d Data0) ToArray() []float64 {
	return []float64{*d.value}
}

func (d Data0) ForEach(f func(index []int, value float64)) {
	f([]int{}, *d.value)
}
