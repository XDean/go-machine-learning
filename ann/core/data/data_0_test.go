package data

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData0(t *testing.T) {
	testData0(t, NewData0)
	testData0_Panic(t, NewData0)
}

func testData0(t *testing.T, df func() Data) {
	d := df()
	d.SetValue(1, nil)
	assert.Equal(t, 1.0, d.GetValue(nil))
	assert.Equal(t, d, d.GetData(nil))

	assert.Equal(t, []int{}, d.GetSize())
	assert.Equal(t, 1, d.GetCount())
	assert.Equal(t, 0, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue(nil))

	assert.Equal(t, []float64{100}, d.ToArray())

	hit := 0
	d.ForEachIndex(func(index []int, value float64) {
		hit++
		assert.Equal(t, []int{}, index)
		assert.Equal(t, 100.0, value)
	})
	assert.Equal(t, 1, hit)

	hit = 0
	d.MapIndex(func(index []int, value float64) float64 {
		hit++
		assert.Equal(t, []int{}, index)
		assert.Equal(t, 100.0, value)
		return 1.0
	})
	assert.Equal(t, 1, hit)
	d.ForEachIndex(func(index []int, value float64) {
		assert.Equal(t, 1.0, value)
	})
}

func testData0_Panic(t *testing.T, df func() Data) {
	d := df()
	assert.Panics(t, func() {
		d.SetValue(1, []int{1})
	})
	assert.Panics(t, func() {
		d.GetValue([]int{1})
	})
	assert.Panics(t, func() {
		d.GetData([]int{1})
	})
}
