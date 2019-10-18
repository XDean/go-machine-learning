package data

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData3(t *testing.T) {
	testData3(t, NewData3)
	testData3_Panic(t, NewData3)
}

func testData3(t *testing.T, df func(int, int, int) Data) {
	d := df(3, 5, 2)
	d.SetValue(1, []int{2, 2, 1})
	assert.Equal(t, 1.0, d.GetValue([]int{2, 2, 1}))
	assert.Equal(t, 0.0, d.GetValue([]int{1, 1, 0}))
	assert.Equal(t, d, d.GetData(nil))
	assert.Equal(t, 1.0, d.GetData([]int{2}).GetValue([]int{2, 1}))
	assert.Equal(t, 1.0, d.GetData([]int{2, 2}).GetValue([]int{1}))
	assert.Equal(t, 1.0, d.GetData([]int{2, 2, 1}).GetValue(nil))
	d.GetData([]int{2, 1}).SetValue(3.0, []int{1})
	assert.Equal(t, 3.0, d.GetValue([]int{2, 1, 1}))

	assert.Equal(t, []int{3, 5, 2}, d.GetSize())
	assert.Equal(t, 30, d.GetCount())
	assert.Equal(t, 3, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue([]int{1, 1, 1}))
	assert.Equal(t, 100.0, d.GetValue([]int{2, 2, 1}))

	assert.Equal(t, []float64{100, 100, 100}, d.ToArray()[:3])

	hit := 0
	d.ForEachIndex(func(index []int, value float64) {
		hit++
		assert.Equal(t, 3, len(index))
		assert.Equal(t, 100.0, value)
	})
	assert.Equal(t, 30, hit)

	hit = 0
	d.Map(func(value float64) float64 {
		return 5
	})
	d.MapIndex(func(index []int, value float64) float64 {
		hit++
		assert.Equal(t, 3, len(index))
		assert.Equal(t, 5.0, value)
		return 1.0
	})
	assert.Equal(t, 30, hit)
	d.ForEachIndex(func(index []int, value float64) {
		assert.Equal(t, 1.0, value)
	})
	d.ForEach(func(value float64) {
		assert.Equal(t, 1.0, value)
	})
}

func testData3_Panic(t *testing.T, df func(int, int, int) Data) {
	d := df(10, 10, 10)
	assert.Panics(t, func() {
		d.SetValue(1, nil)
	})
	assert.Panics(t, func() {
		d.GetValue(nil)
	})
	assert.Panics(t, func() {
		d.GetValue([]int{100, 100, 100})
	})
	assert.Panics(t, func() {
		d.GetData([]int{1, 1, 1, 1})
	})
	assert.Panics(t, func() {
		d.GetData([]int{100})
	})
}
