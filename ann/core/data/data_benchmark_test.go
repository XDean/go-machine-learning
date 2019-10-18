package data

import "testing"

func BenchmarkData2_New(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewData2(1000, 1000)
	}
}

func BenchmarkDataN_New(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewDataN([]int{1000, 1000})
	}
}

func BenchmarkData2_Fill(b *testing.B) {
	d := NewData2(1000, 1000)
	for i := 0; i < b.N; i++ {
		d.Fill(float64(i))
	}
}

func BenchmarkDataN_Fill(b *testing.B) {
	d := NewDataN([]int{1000, 1000})
	for i := 0; i < b.N; i++ {
		d.Fill(float64(i))
	}
}

func BenchmarkSlice_Fill(b *testing.B) {
	s := [1000][1000]float64{}
	for i := 0; i < b.N; i++ {
		for i, v := range s {
			for j, _ := range v {
				s[i][j] = float64(i)
			}
		}
	}
}

func BenchmarkData2_ForEach(b *testing.B) {
	d := NewData2(1000, 1000)
	for i := 0; i < b.N; i++ {
		d.ForEachIndex(func(index []int, value float64) {
			_ = len(index)
		})
	}
}

func BenchmarkDataN_ForEach(b *testing.B) {
	d := NewDataN([]int{1000, 1000})
	for i := 0; i < b.N; i++ {
		d.ForEachIndex(func(index []int, value float64) {
			_ = len(index)
		})
	}
}

func BenchmarkSlice_ForEach(b *testing.B) {
	s := [1000][1000]float64{}
	for i := 0; i < b.N; i++ {
		for i, v := range s {
			for j, _ := range v {
				_ = i * j
			}
		}
	}
}

func BenchmarkData2_Map(b *testing.B) {
	d := NewData2(1000, 1000)
	for n := 0; n < b.N; n++ {
		d.MapIndex(func(index []int, value float64) float64 {
			return float64(index[0] + index[1] + n)
		})
	}
}

func BenchmarkDataN_Map(b *testing.B) {
	d := NewDataN([]int{1000, 1000})
	for n := 0; n < b.N; n++ {
		d.MapIndex(func(index []int, value float64) float64 {
			return float64(index[0] + index[1] + n)
		})
	}
}

func BenchmarkSlice_Map(b *testing.B) {
	s := [1000][1000]float64{}
	for n := 0; n < b.N; n++ {
		for i, v := range s {
			for j, _ := range v {
				s[i][j] = float64(i + j + n)
			}
		}
	}
}
