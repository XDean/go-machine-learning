package mnist

import (
	"encoding/binary"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/util"
	"image"
	"image/color"
	"io"
	"os"
)

type (
	MnistData struct {
		Image []byte
		Label uint8
	}
)

func MnistLoad(imageFile, labelFile string) <-chan MnistData {
	result := make(chan MnistData)
	go func() {
		defer close(result)
		images, err := os.Open(imageFile)
		util.NoError(err)
		labels, err := os.Open(labelFile)
		util.NoError(err)

		imageChan := readImageFile(images)
		labelChan := readLabelFile(labels)
		for {
			i, iok := <-imageChan
			l, lok := <-labelChan
			if iok == lok {
				if iok {
					result <- MnistData{
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

func (m MnistData) ColorModel() color.Model {
	return color.GrayModel
}

func (m MnistData) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: 0, Y: 0},
		Max: image.Point{X: 28, Y: 28},
	}
}

func (m MnistData) At(x, y int) color.Color {
	return color.Gray{Y: m.Image[y*28+x]}
}
