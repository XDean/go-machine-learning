package data

import (
	"testing"
)

func TestDataReflect(t *testing.T) {
	testData0(t, func() Data {
		return NewDataReflect(nil)
	})
	testData0_Panic(t, func() Data {
		return NewDataReflect(nil)
	})
	testData1(t, func(i int) Data {
		return NewDataReflect([]int{i})
	})
	testData1_Panic(t, func(i int) Data {
		return NewDataReflect([]int{i})
	})
	testData2(t, func(i, j int) Data {
		return NewDataReflect([]int{i, j})
	})
	testData2_Panic(t, func(i, j int) Data {
		return NewDataReflect([]int{i, j})
	})
	testData3(t, func(i, j, k int) Data {
		return NewDataReflect([]int{i, j, k})
	})
	testData3_Panic(t, func(i, j, k int) Data {
		return NewDataReflect([]int{i, j, k})
	})
}
