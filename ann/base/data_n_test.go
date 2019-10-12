package base

import (
	"testing"
)

func TestDataN(t *testing.T) {
	testData0(t, func() Data {
		return NewDataN()
	})
	testData0_Panic(t, func() Data {
		return NewDataN()
	})
	testData1(t, func(i int) Data {
		return NewDataN(i)
	})
	testData1_Panic(t, func(i int) Data {
		return NewDataN(i)
	})
	testData2(t, func(i, j int) Data {
		return NewDataN(i, j)
	})
	testData2_Panic(t, func(i, j int) Data {
		return NewDataN(i, j)
	})
	testData3(t, func(i, j, k int) Data {
		return NewDataN(i, j, k)
	})
	testData3_Panic(t, func(i, j, k int) Data {
		return NewDataN(i, j, k)
	})
}
