package model

func SizeToCount(size ...uint) uint {
	count := uint(1)
	for _, v := range size {
		count *= v
	}
	return count
}
