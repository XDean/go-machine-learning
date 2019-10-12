package base

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestData_IsLeaf(t *testing.T) {
	assert.True(t, NewData().IsLeaf())
	assert.False(t, NewData(1).IsLeaf())
}

func TestData_GetValue(t *testing.T) {
	d := NewData(2, 2, 2)
	assert.Equal(t, 0.0, d.GetValue(1, 1, 0))
	assert.Equal(t, 0.0, d.GetValue(1, 0, 1))
	d = d.SetValue(0.5, 1, 1, 0)
	assert.Equal(t, 0.5, d.GetValue(1, 1, 0))
	assert.Equal(t, 0.0, d.GetValue(1, 0, 1))

	assert.Panics(t, func() {
		d.GetValue(1, 1)
	})
	assert.Panics(t, func() {
		d.GetValue(0, 0, 3)
	})
	assert.Panics(t, func() {
		d.SetValue(1, 1)
	})
	assert.Panics(t, func() {
		d.SetValue(0, 0, 3)
	})
}

func TestData_GetData(t *testing.T) {
	d := NewData(2, 2, 2)

	s1 := d.GetData(1, 1)
	assert.Equal(t, 0.0, s1.GetValue(0))

	s1 = s1.SetValue(0.5, 0)
	assert.Equal(t, 0.5, s1.GetValue(0))

	assert.Panics(t, func() {
		d.GetData(3)
	})
}

func TestData_GetSize(t *testing.T) {
	assert.Equal(t, []uint{}, NewData().GetSize())
	assert.Equal(t, []uint{2, 3, 4}, NewData(2, 3, 4).GetSize())
}

func TestData_GetCount(t *testing.T) {
	assert.Equal(t, uint(1), NewData().GetCount())
	assert.Equal(t, uint(24), NewData(2, 3, 4).GetCount())
}

func TestData_GetDim(t *testing.T) {
	assert.Equal(t, uint(0), NewData().GetDim())
	assert.Equal(t, uint(3), NewData(2, 3, 4).GetDim())
}

func TestData_ToDim(t *testing.T) {
	d0t5 := NewData().ToDim(5)
	assert.Equal(t, []uint{1, 1, 1, 1, 1}, d0t5.GetSize())
	assert.Equal(t, uint(1), d0t5.GetCount())
	assert.Equal(t, uint(5), d0t5.GetDim())

	d3t2 := NewData(2, 3, 4).ToDim(2)
	assert.Equal(t, []uint{2, 12}, d3t2.GetSize())
	assert.Equal(t, uint(24), d3t2.GetCount())
	assert.Equal(t, uint(2), d3t2.GetDim())

	assert.Equal(t, d3t2, d3t2.ToDim(2))

	assert.Panics(t, func() {
		d3t2.ToDim(0)
	})
}

func TestData_Fill(t *testing.T) {
	assert.Equal(t, 5.0, NewData().Fill(5).GetValue())
	assert.Equal(t, []float64{5, 5, 5, 5}, NewData(2, 2).Fill(5).ToArray())
}

func TestData_Identity2D(t *testing.T) {
	assert.Equal(t, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1}, NewData(2, 2).Identity2D().ToArray())
	assert.Equal(t, []float64{1, 0, 0, 1}, NewData(2).Identity2D().ToArray())
}

func TestData_ToArray(t *testing.T) {
	assert.Equal(t, []float64{1, 1, 1, 1}, NewData(2, 2).Fill(1).ToArray())
	assert.Equal(t, []float64{1}, NewData().Fill(1).ToArray())
}
