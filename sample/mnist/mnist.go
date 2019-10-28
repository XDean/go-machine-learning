package mnist

import (
	"encoding/binary"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/core/util"
	"io"
	"math"
	"os"
)

func Load(imageFile, labelFile string, limit int) <-chan Data {
	if limit <= 0 {
		limit = math.MaxInt32
	}
	result := make(chan Data, 10)
	go func() {
		defer close(result)
		images, err := os.Open(imageFile)
		util.NoError(err)
		labels, err := os.Open(labelFile)
		util.NoError(err)

		imageChan := readImageFile(images)
		labelChan := readLabelFile(labels)
		for ; limit > 0; limit-- {
			i, iok := <-imageChan
			l, lok := <-labelChan
			if iok == lok {
				if iok {
					result <- Data{
						Image: i,
						Label: l,
					}
				} else {
					break
				}
			} else {
				panic("MNIST image and label are not sync")
			}
		}
	}()
	return result
}

func readImageFile(r io.Reader) <-chan []byte {
	const imageMagic = 0x00000803
	var (
		magic int32
		count int32
		row   int32
		col   int32
	)
	util.NoError(binary.Read(r, binary.BigEndian, &magic))
	if magic != imageMagic {
		panic("unrecognized MNIST image data")
	}
	util.NoError(binary.Read(r, binary.BigEndian, &count))
	util.NoError(binary.Read(r, binary.BigEndian, &row))
	util.NoError(binary.Read(r, binary.BigEndian, &col))
	if row != 28 || col != 28 {
		panic(fmt.Sprintf("unrecognized MNIST image data. Except 28x28, but %dx%d", row, col))
	}
	pixels := int(row * col)
	result := make(chan []byte, 10)
	go func() {
		defer close(result)
		for i := 0; i < int(count); i++ {
			buffer := make([]byte, pixels)
			read, err := io.ReadFull(r, buffer)
			util.NoError(err)
			if read != pixels {
				panic(io.EOF)
			}
			result <- buffer
		}
	}()
	return result
}

func readLabelFile(r io.Reader) <-chan uint8 {
	const labelMagic = 0x00000801

	var magic int32
	util.NoError(binary.Read(r, binary.BigEndian, &magic))
	if magic != labelMagic {
		panic("unrecognized MNIST label data")
	}

	var count int32
	util.NoError(binary.Read(r, binary.BigEndian, &count))

	result := make(chan uint8, 5)
	go func() {
		defer close(result)
		for i := int32(0); i < count; i++ {
			var l uint8
			util.NoError(binary.Read(r, binary.BigEndian, &l))
			result <- l
		}
	}()
	return result
}
