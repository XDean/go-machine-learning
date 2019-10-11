package main

import (
	"errors"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/base"
	"github.com/XDean/go-machine-learning/ann/model"
	"github.com/XDean/go-machine-learning/data/mnist"
	"github.com/urfave/cli"
	"log"
	"os"
	"path/filepath"
)

var (
	loadPath = ""
	savePath = ""
	dataPath = ""
	modelN   = -1
)

const (
	train_image = "t10k-images.idx3-ubyte"
	train_label = "t10k-labels.idx1-ubyte"
	test_image  = "train-images.idx3-ubyte"
	test_label  = "train-labels.idx1-ubyte"
)

func main() {
	app := cli.NewApp()

	app.Name = "XDean Go ANN Sample"

	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:        "load",
			Usage:       "Path to load model",
			Destination: &loadPath,
		},
		cli.StringFlag{
			Name:        "data",
			Usage:       fmt.Sprintf("Folder contains MNIST data. It must contains 4 files: [%s, %s, %s, %s]", train_image, train_label, test_image, test_label),
			Destination: &dataPath,
		},
	}

	app.Commands = []cli.Command{
		{
			Name:   "train",
			Action: Train,
			Flags: []cli.Flag{
				cli.IntFlag{
					Name: "model",
					Usage: "Use an new built-in model to train. Input the model number. " +
						"You can use `show` command to see available model. Note it will be override by model from loadPath.",
					Destination: &modelN,
				},
				cli.StringFlag{
					Name:        "save",
					Usage:       "Path to save model. If empty, save to load path.",
					Destination: &savePath,
				},
			},

			Subcommands: []cli.Command{
				{
					Name:   "show",
					Usage:  "Show available models",
					Action: Show,
				},
			},
		},
		{
			Name:   "test",
			Action: Test,
		},
		{
			Name:   "predict",
			Action: Predict,
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}

var models = make([]*model.Model, 0)

func RegisterModel(m *model.Model) {
	m.Init()
	models = append(models, m)
}

func Show(c *cli.Context) error {
	for i, v := range models {
		fmt.Printf("%d. %s\n", i+1, v.Name)
	}
	return nil
}

func Train(c *cli.Context) (err error) {
	defer base.RecoverNoError(&err)
	checkData()

	m, err := loadModel()
	base.NoError(err)

	datas := mnist.MnistLoad(filepath.Join(dataPath, train_image), filepath.Join(dataPath, train_label))
	count := 0
	for {
		data, ok := <-datas
		if !ok {
			break
		}
		count++
		result := m.Feed(mnistToData(data))
		fmt.Printf("%5d: expect %d, predict %d, error %.2f\n", count, data.Label, predictFromResult(result), result.TotalError)
	}
	return saveModel(m)
}

func Test(c *cli.Context) (err error) {
	defer base.RecoverNoError(&err)
	checkData()

	m, err := loadModel()
	base.NoError(err)

	datas := mnist.MnistLoad(filepath.Join(dataPath, test_image), filepath.Join(dataPath, test_label))

	count := 0
	correct := 0
	for {
		data, ok := <-datas
		if !ok {
			break
		}
		count++
		result := m.Test(mnistToData(data))
		predict := predictFromResult(result)
		if predict == uint(data.Label) {
			correct++
		}
		fmt.Printf("%5d: expect %d, predict %d, error rate %.2f%%\n", count, data.Label, predict, 100-float64(correct)/float64(count)*100)
	}
	return
}

func Predict(c *cli.Context) error {
	// TODO
	return nil
}

func checkData() {
	if dataPath == "" {
		panic("Data path not specified")
	}
}

func loadModel() (result *model.Model, err error) {
	if modelN > 0 && modelN <= len(models) {
		result = models[modelN-1]
		fmt.Printf("New model: %s\n", result.Name)
	}
	if loadPath != "" {
		fmt.Printf("Load model from %s\n", loadPath)
		result = new(model.Model)
		err = result.LoadFromFile(loadPath)
	}
	if result == nil {
		err = errors.New("Model is not specified.")
	}
	return
}

func saveModel(m *model.Model) (err error) {
	if loadPath == "" {
		loadPath = savePath
	}
	if loadPath != "" {
		fmt.Printf("Save model to %s\n", loadPath)
		err = m.SaveToFile(loadPath)
	} else {
		return errors.New("Save path not spcified.")
	}
	return
}

func mnistToData(d mnist.MnistData) (input, target base.Data) {
	input = base.NewData(28, 28)
	target = base.NewData(10)

	input.ForEach(func(index []uint, value float64) {
		input.SetValue(float64(d.Image[index[0]*28+index[1]]), index...)
	})
	target.SetValue(1, uint(d.Label))
	return
}

func predictFromResult(r model.Result) uint {
	max := uint(0)
	r.Output.ForEach(func(index []uint, value float64) {
		if value > r.Output.GetValue(max) {
			max = index[0]
		}
	})
	return max
}
