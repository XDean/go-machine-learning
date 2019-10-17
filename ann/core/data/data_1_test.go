package data

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData1(t *testing.T) {
	testData1(t, NewData1)
	testData1_Panic(t, NewData1)
}

func testData1(t *testing.T, df func(int) Data) {
	d := df(3)
	d.SetValue(1, 2)
	assert.Equal(t, 1.0, d.GetValue(2))
	assert.Equal(t, 0.0, d.GetValue(1))
	assert.Equal(t, d, d.GetData())
	assert.Equal(t, 1.0, d.GetData(2).GetValue())

	assert.Equal(t, []int{3}, d.GetSize())
	assert.Equal(t, 3, d.GetCount())
	assert.Equal(t, 1, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue(1))
	assert.Equal(t, 100.0, d.GetValue(2))

	assert.Equal(t, []float64{100, 100, 100}, d.ToArray())

	hit := 0
	d.ForEachIndex(func(index []int, value float64) {
		hit++
		assert.Equal(t, 1, len(index))
		assert.Equal(t, 100.0, value)
	})
	assert.Equal(t, 3, hit)

	hit = 0
	d.MapIndex(func(index []int, value float64) float64 {
		hit++
		assert.Equal(t, 1, len(index))
		assert.Equal(t, 100.0, value)
		return 1.0
	})
	assert.Equal(t, 3, hit)
	d.ForEachIndex(func(index []int, value float64) {
		assert.Equal(t, 1.0, value)
	})
}

func testData1_Panic(t *testing.T, df func(int) Data) {
	d := df(10)
	assert.Panics(t, func() {
		d.SetValue(1)
	})
	assert.Panics(t, func() {
		d.GetValue()
	})
	assert.Panics(t, func() {
		d.GetValue(100)
	})
	assert.Panics(t, func() {
		d.GetData(1, 2)
	})
	assert.Panics(t, func() {
		d.GetData(100)
	})
}
