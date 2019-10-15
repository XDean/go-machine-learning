package data

import (
	"testing"
)

func TestDataRecursive(t *testing.T) {
	testData0(t, func() Data {
		return NewDataRecusive()
	})
	testData0_Panic(t, func() Data {
		return NewDataRecusive()
	})
	testData1(t, func(i int) Data {
		return NewDataRecusive(i)
	})
	testData1_Panic(t, func(i int) Data {
		return NewDataRecusive(i)
	})
	testData2(t, func(i, j int) Data {
		return NewDataRecusive(i, j)
	})
	testData2_Panic(t, func(i, j int) Data {
		return NewDataRecusive(i, j)
	})
	testData3(t, func(i, j, k int) Data {
		return NewDataRecusive(i, j, k)
	})
	testData3_Panic(t, func(i, j, k int) Data {
		return NewDataRecusive(i, j, k)
	})
}
