package data

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
)

type Data0 struct {
	Value *float64
}

func NewData0() Data {
	value := 0.0
	return Data0{Value: &value}
}

func (d Data0) SetValue(value float64, indexes ...int) Data {
	util.MustTrue(len(indexes) == 0)
	*d.Value = value
	return d
}

func (d Data0) GetValue(indexes ...int) float64 {
	util.MustTrue(len(indexes) == 0)
	return *d.Value
}

func (d Data0) GetData(indexes ...int) Data {
	util.MustTrue(len(indexes) == 0)
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
	*d.Value = value
	return d
}

func (d Data0) ToArray() []float64 {
	return []float64{*d.Value}
}

func (d Data0) ForEach(f func(index []int, value float64)) {
	f([]int{}, *d.Value)
}

func (d Data0) Map(f func(index []int, value float64) float64) {
	*d.Value = f([]int{}, *d.Value)
}
