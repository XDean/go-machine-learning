package data

import "fmt"

func checkIndex(size []int, indexes []int, match bool) error {
	if match && len(indexes) != len(size) {
		return fmt.Errorf("Index not match, actual %d, get %d", len(size), len(indexes))
	}
	for i, v := range indexes {
		if v >= size[i] {
			return fmt.Errorf("Index out of bound: actual %v, get %v", size, indexes)
		}
	}
	return nil
}

func indexesToIndex(size []int, indexes []int) int {
	index := 0
	for i, v := range indexes {
		index = size[i]*index + v
	}
	return index
}

func indexToIndexes(size []int, index int) []int {
	result := make([]int, len(size))
	for i := range size {
		s := size[len(size)-1-i]
		left := index % s
		index = (index - left) / s
		result[len(size)-1-i] = left
	}
	return result
}

func forIndex(size []int, f func(index int, indexes []int)) {
	len := len(size)
	index := 0
	indexes := make([]int, len)
	for {
		f(index, indexes)
		index++
		add := len - 1
		for add >= 0 {
			if indexes[add] == size[add]-1 {
				indexes[add] = 0
				add--
			} else {
				indexes[add]++
				break
			}
		}
		if add < 0 {
			break
		}
	}
}

func ForEachSub(d Data, f func(indexes []int, value float64), subIndex ...int) {
	d.ForEach(func(indexes []int, value float64) {
		for i, v := range subIndex {
			if indexes[i] != v {
				return
			}
			f(indexes[len(subIndex):], value)
		}
	})
}

func MapSub(d Data, f func(indexes []int, value float64) float64, subIndex ...int) {
	d.Map(func(indexes []int, value float64) float64 {
		for i, v := range subIndex {
			if indexes[i] != v {
				return value
			}
			return f(indexes[len(subIndex):], value)
		}
	})
}
