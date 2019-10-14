package base

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
	d.SetValue(1, 2, 2, 1)
	assert.Equal(t, 1.0, d.GetValue(2, 2, 1))
	assert.Equal(t, 0.0, d.GetValue(1, 1, 0))
	assert.Equal(t, d, d.GetData())
	assert.Equal(t, 1.0, d.GetData(2).GetValue(2, 1))
	assert.Equal(t, 1.0, d.GetData(2, 2).GetValue(1))
	assert.Equal(t, 1.0, d.GetData(2, 2, 1).GetValue())

	assert.Equal(t, []int{3, 5, 2}, d.GetSize())
	assert.Equal(t, 30, d.GetCount())
	assert.Equal(t, 3, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue(1, 1, 1))
	assert.Equal(t, 100.0, d.GetValue(2, 2, 1))

	assert.Equal(t, []float64{100, 100, 100}, d.ToArray()[:3])

	hit := 0
	d.ForEach(func(index []int, value float64) {
		hit++
		assert.Equal(t, 3, len(index))
		assert.Equal(t, 100.0, value)
	})
	assert.Equal(t, 30, hit)
}

func testData3_Panic(t *testing.T, df func(int, int, int) Data) {
	d := df(10, 10, 10)
	assert.Panics(t, func() {
		d.SetValue(1)
	})
	assert.Panics(t, func() {
		d.GetValue()
	})
	assert.Panics(t, func() {
		d.GetValue(100, 100, 100)
	})
	assert.Panics(t, func() {
		d.GetData(1, 1, 1, 1)
	})
	assert.Panics(t, func() {
		d.GetData(100)
	})
}
