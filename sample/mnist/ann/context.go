package main

import (
	"errors"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/util"
	"github.com/XDean/go-machine-learning/sample/mnist/data/mnist"
	"path/filepath"
	"time"
)

type Context struct {
	loadPath string
	savePath string
	dataPath string
	limit    int
	modelN   int
	repeat   int
}

func (c Context) Show() error {
	for i, v := range models {
		fmt.Printf("%d. %s\n", i+1, v.Name)
	}
	return nil
}

func (c Context) Train() (err error) {
	defer util.RecoverNoError(&err)
	c.checkData()

	m, err := c.loadModel()
	util.NoError(err)
	m.Init()

	datas := mnist.Load(filepath.Join(c.dataPath, train_image), filepath.Join(c.dataPath, train_label), c.limit)
	profile := NewProfile(100)
	for {
		data, ok := <-datas
		if !ok {
			break
		}
		input, target := mnistToData(data)
		result := m.FeedTimes(input, target, c.repeat)[0]
		predict := predictFromResult(result)
		profile.Add(data.Label == uint8(predict))
		fmt.Printf("%5d: expect %d, predict %d, error %.4f, correct %.2f%%, recent %.2f%%, time %d ms\n",
			profile.Total, data.Label, predict, result.TotalError, profile.HitRate()*100, profile.RecentHitRate()*100, result.Time/time.Millisecond)
	}
	return c.saveModel(m)
}

func (c Context) Test() (err error) {
	defer util.RecoverNoError(&err)
	c.checkData()

	m, err := c.loadModel()
	util.NoError(err)
	m.Init()

	datas := mnist.Load(filepath.Join(c.dataPath, test_image), filepath.Join(c.dataPath, test_label), c.limit)

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
		if predict == int(data.Label) {
			correct++
		}
		fmt.Printf("%5d: expect %d, predict %d, error rate %.4f%%\n", count, data.Label, predict, 100-float64(correct)/float64(count)*100)
	}
	return
}

func (c Context) Predict() error {
	// TODO
	return nil
}

func (c Context) checkData() {
	if c.dataPath == "" {
		panic("Data path not specified")
	}
}

func (c Context) loadModel() (result *model.Model, err error) {
	if c.modelN > 0 && c.modelN <= len(models) {
		result = models[c.modelN-1]
		fmt.Printf("New model: %s\n", result.Name)
	}
	if c.loadPath != "" {
		fmt.Printf("Load model from %s\n", c.loadPath)
		result = new(model.Model)
		err = result.LoadFromFile(c.loadPath)
	}
	if result == nil {
		err = errors.New("Model is not specified.")
	}
	return
}

func (c Context) saveModel(m *model.Model) (err error) {
	if c.loadPath == "" {
		c.loadPath = c.savePath
	}
	if c.loadPath != "" {
		fmt.Printf("Save model to %s\n", c.loadPath)
		err = m.SaveToFile(c.loadPath)
	} else {
		return errors.New("Save path not spcified.")
	}
	return
}
