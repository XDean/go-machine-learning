package data

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData2(t *testing.T) {
	testData2(t, NewData2)
	testData2_Panic(t, NewData2)
}

func testData2(t *testing.T, df func(int, int) Data) {
	d := df(3, 5)
	d.SetValue(1, 2, 2)
	assert.Equal(t, 1.0, d.GetValue(2, 2))
	assert.Equal(t, 0.0, d.GetValue(1, 1))
	assert.Equal(t, d, d.GetData())
	assert.Equal(t, 1.0, d.GetData(2).GetValue(2))
	assert.Equal(t, 1.0, d.GetData(2, 2).GetValue())

	assert.Equal(t, []int{3, 5}, d.GetSize())
	assert.Equal(t, 15, d.GetCount())
	assert.Equal(t, 2, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue(1, 1))
	assert.Equal(t, 100.0, d.GetValue(2, 2))

	assert.Equal(t, []float64{100, 100, 100}, d.ToArray()[:3])

	hit := 0
	d.ForEachIndex(func(index []int, value float64) {
		hit++
		assert.Equal(t, 2, len(index))
		assert.Equal(t, 100.0, value)
	})
	assert.Equal(t, 15, hit)

	hit = 0
	d.MapIndex(func(index []int, value float64) float64 {
		hit++
		assert.Equal(t, 2, len(index))
		assert.Equal(t, 100.0, value)
		return 1.0
	})
	assert.Equal(t, 15, hit)
	d.ForEachIndex(func(index []int, value float64) {
		assert.Equal(t, 1.0, value)
	})
}

func testData2_Panic(t *testing.T, df func(int, int) Data) {
	d := df(10, 10)
	assert.Panics(t, func() {
		d.SetValue(1)
	})
	assert.Panics(t, func() {
		d.GetValue()
	})
	assert.Panics(t, func() {
		d.GetValue(100, 100)
	})
	assert.Panics(t, func() {
		d.GetData(1, 1, 1)
	})
	assert.Panics(t, func() {
		d.GetData(100)
	})
}
