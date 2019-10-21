package mnist

import (
	"github.com/XDean/go-machine-learning/ann/core/util"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
)

type (
	Data struct {
		Image []byte
		Label uint8
	}
)

func (m Data) ColorModel() color.Model {
	return color.GrayModel
}

func (m Data) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: 0, Y: 0},
		Max: image.Point{X: 28, Y: 28},
	}
}

func (m Data) At(x, y int) color.Color {
	return color.Gray{Y: m.Image[y*28+x]}
}

func (m Data) Save(w io.Writer) error {
	return png.Encode(w, m)
}

func (m Data) SaveToFile(file string) (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(os.MkdirAll(filepath.Dir(file), 0755))
	f, err := os.Create(file)
	util.NoError(err)
	defer f.Close()
	return m.Save(f)
}
