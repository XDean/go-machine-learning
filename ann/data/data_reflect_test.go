package data

import (
	"testing"
)

func TestDataReflect(t *testing.T) {
	testData0(t, func() Data {
		return NewDataReflect()
	})
	testData0_Panic(t, func() Data {
		return NewDataReflect()
	})
	testData1(t, func(i int) Data {
		return NewDataReflect(i)
	})
	testData1_Panic(t, func(i int) Data {
		return NewDataReflect(i)
	})
	testData2(t, func(i, j int) Data {
		return NewDataReflect(i, j)
	})
	testData2_Panic(t, func(i, j int) Data {
		return NewDataReflect(i, j)
	})
	testData3(t, func(i, j, k int) Data {
		return NewDataReflect(i, j, k)
	})
	testData3_Panic(t, func(i, j, k int) Data {
		return NewDataReflect(i, j, k)
	})
}
