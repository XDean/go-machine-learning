package model

func SizeToCount(size ...uint) uint {
	count := uint(1)
	for _, v := range size {
		count *= v
	}
	return count
}

func ErrorToInput(l Layer) Data {
	errorToOutput := l.GetErrorToOutput()
	outputToInput := l.GetOutputToInput()

	os := errorToOutput.GetSize()
	ois := outputToInput.GetSize()
	size := ois[len(os):]

	errorToInput := NewData(size...)
	errorToInput.ForEach(func(inputIndex []uint, value float64) {
		sum := 0.0
		errorToOutput.ForEach(func(outputIndex []uint, value float64) {
			sum += value * outputToInput.GetValue(append(outputIndex, inputIndex...)...)
		})
		errorToInput.SetValue(sum, inputIndex...)
	})
	return errorToInput
}
