package base

import (
	"bytes"
	"github.com/XDean/go-machine-learning/ann/persistent"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestToDim(t *testing.T) {
	d0t5 := ToDim(NewData(), 5)
	assert.Equal(t, []int{1, 1, 1, 1, 1}, d0t5.GetSize())
	assert.Equal(t, int(1), d0t5.GetCount())
	assert.Equal(t, int(5), d0t5.GetDim())

	d2t1 := ToDim(NewData(3, 3), 1)
	assert.Equal(t, []int{9}, d2t1.GetSize())
	assert.Equal(t, int(9), d2t1.GetCount())
	assert.Equal(t, int(1), d2t1.GetDim())

	d3t2 := ToDim(NewData(2, 3, 4), 2)
	assert.Equal(t, []int{2, 12}, d3t2.GetSize())
	assert.Equal(t, int(24), d3t2.GetCount())
	assert.Equal(t, int(2), d3t2.GetDim())

	assert.Equal(t, d3t2, ToDim(d3t2, 2))

	assert.Panics(t, func() {
		ToDim(d3t2, 0)
	})
}

func TestData_Identity2D(t *testing.T) {
	assert.Equal(t, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1}, Identity2D(NewData(2, 2)).ToArray())
	assert.Equal(t, []float64{1, 0, 0, 1}, Identity2D(NewData(2)).ToArray())
}

func testSaveLoad(t *testing.T, data Data) {
	buffer := new(bytes.Buffer)

	encoder := persistent.NewEncoder(buffer)
	assert.NoError(t, encoder.Encode(&data))

	var newData Data

	decoder := persistent.NewDecoder(buffer)
	assert.NoError(t, decoder.Decode(&newData))

	assert.Equal(t, data.GetSize(), newData.GetSize())
	assert.Equal(t, data.GetCount(), newData.GetCount())
	assert.Equal(t, data.GetDim(), newData.GetDim())
	assert.Equal(t, data.ToArray(), newData.ToArray())
}
