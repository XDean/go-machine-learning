package main

import (
	"fmt"
	core2 "github.com/XDean/go-machine-learning/ann/core"
	"github.com/urfave/cli"
	"log"
	"math/rand"
	"os"
	"time"
)

const (
	test_image  = "t10k-images.idx3-ubyte"
	test_label  = "t10k-labels.idx1-ubyte"
	train_image = "train-images.idx3-ubyte"
	train_label = "train-labels.idx1-ubyte"
)

func main() {
	rand.Seed(time.Now().Unix())
	app := cli.NewApp()

	app.Name = "XDean Go ANN Sample"

	ctx := Context{}

	loadFlag := cli.StringFlag{
		Name:        "load",
		Usage:       "Path to load model",
		Destination: &ctx.loadPath,
	}
	dataFlag := cli.StringFlag{
		Name:        "data",
		Usage:       fmt.Sprintf("Folder contains MNIST data. It must contains 4 files: [%s, %s, %s, %s]", train_image, train_label, test_image, test_label),
		Destination: &ctx.dataPath,
	}
	limitFlag := cli.IntFlag{
		Name:        "limit",
		Usage:       "Limit data count",
		Destination: &ctx.limit,
	}

	app.Commands = []cli.Command{
		{
			Name:   "train",
			Action: func(*cli.Context) error { return ctx.Train() },
			Flags: []cli.Flag{
				cli.IntFlag{
					Name: "model",
					Usage: "Use an new built-in model to train. Input the model number. " +
						"You can use 'show' command to see available model. Note it will be override by model from loadPath.",
					Destination: &ctx.modelN,
				},
				cli.StringFlag{
					Name:        "save",
					Usage:       "Path to save model. If empty, save to load path.",
					Destination: &ctx.savePath,
				},
				cli.IntFlag{
					Name:        "batch",
					Usage:       "Batch size, default 1",
					Destination: &ctx.batch,
					Value:       1,
				},
				loadFlag,
				dataFlag,
				limitFlag,
			},

			Subcommands: []cli.Command{
				{
					Name:   "show",
					Usage:  "Show available models",
					Action: func(*cli.Context) error { return ctx.Show() },
					Flags: []cli.Flag{
						cli.BoolFlag{
							Name:        "full",
							Usage:       "Show Full Model Description",
							Destination: &ctx.fullDesc,
						},
					},
				},
			},
		},
		{
			Name:   "test",
			Action: func(*cli.Context) error { return ctx.Test() },
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:        "export",
					Usage:       "Export failed images to the folder if this value is not empty. By default, don't export.",
					Destination: &ctx.savePath,
				},
				loadFlag,
				dataFlag,
				limitFlag,
			},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}

var models = make([]*core2.Model, 0)

func RegisterModel(m *core2.Model) {
	models = append(models, m)
	m.Init()
}
