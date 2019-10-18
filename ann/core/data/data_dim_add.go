package data

type DimAdd struct {
	Actual Data
	AddDim int
}

func NewDimAdd(actual Data, addDim int) Data {
	return DimAdd{Actual: actual, AddDim: addDim}
}

func (d DimAdd) SetValue(value float64, indexes []int) Data {
	d.Actual.SetValue(value, indexes[:d.Actual.GetDim()])
	return d
}

func (d DimAdd) GetValue(indexes []int) float64 {
	return d.Actual.GetValue(indexes[:d.Actual.GetDim()])
}

func (d DimAdd) GetData(indexes []int) Data {
	if len(indexes) > d.Actual.GetDim() {
		return NewDimAdd(
			NewData0().SetValue(d.GetValue(indexes), nil),
			d.AddDim-len(indexes)+d.Actual.GetDim(),
		)
	} else {
		return NewDimAdd(
			d.GetData(indexes),
			d.AddDim,
		)
	}
}

func (d DimAdd) GetSize() []int {
	result := make([]int, d.GetDim())
	for i := range result {
		result[i] = 1
	}
	copy(result, d.Actual.GetSize())
	return result
}

func (d DimAdd) GetCount() int {
	return d.Actual.GetCount()
}

func (d DimAdd) GetDim() int {
	return d.Actual.GetDim() + d.AddDim
}

func (d DimAdd) Fill(value float64) Data {
	d.Map(func(_ float64) float64 { return value })
	return d
}

func (d DimAdd) ToArray() []float64 {
	return d.Actual.ToArray()
}

func (d DimAdd) ForEachIndex(f func(index []int, value float64)) {
	indexes := make([]int, d.GetDim())
	d.Actual.ForEachIndex(func(actualIndex []int, value float64) {
		copy(indexes, actualIndex)
		f(indexes, value)
	})
}

func (d DimAdd) MapIndex(f func(index []int, value float64) float64) {
	indexes := make([]int, d.GetDim())
	d.Actual.MapIndex(func(actualIndex []int, value float64) float64 {
		copy(indexes, actualIndex)
		return f(indexes, value)
	})
}

func (d DimAdd) ForEach(f func(value float64)) {
	d.Actual.ForEach(f)
}

func (d DimAdd) Map(f func(value float64) float64) {
	d.Actual.Map(f)
}
