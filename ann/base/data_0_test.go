package base

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData0(t *testing.T) {
	d := NewData0()
	d.SetValue(1)
	assert.Equal(t, 1.0, d.GetValue())
	assert.Equal(t, d, d.GetData())

	assert.Equal(t, []int{}, d.GetSize())
	assert.Equal(t, 1, d.GetCount())
	assert.Equal(t, 0, d.GetDim())

	d.Fill(100)
	assert.Equal(t, 100.0, d.GetValue())

	assert.Equal(t, []float64{100}, d.ToArray())

	d.ForEach(func(index []int, value float64) {
		assert.Equal(t, []int{}, index)
		assert.Equal(t, 100.0, value)
	})
}

func TestData0_Panic(t *testing.T) {
	d := NewData0()
	assert.Panics(t, func() {
		d.SetValue(1, 1)
	})
	assert.Panics(t, func() {
		d.GetValue(1)
	})
	assert.Panics(t, func() {
		d.GetData(1)
	})
}
