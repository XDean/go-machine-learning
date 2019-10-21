package main

import (
	"errors"
	"fmt"
	"github.com/XDean/go-machine-learning/ann/core/model"
	"github.com/XDean/go-machine-learning/ann/core/util"
	"github.com/XDean/go-machine-learning/sample/mnist"
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
	util.NoError(c.checkData())

	m, err := c.loadModel()
	util.NoError(err)

	datas := mnist.Load(filepath.Join(c.dataPath, train_image), filepath.Join(c.dataPath, train_label), c.limit)
	profile := NewProfile(100)
	for {
		data, ok := <-datas
		if !ok {
			break
		}
		input, target := mnistToData(data)
		var result model.Result
		var predict int
		var usedTime int64
		if c.repeat > 0 {
			results := m.FeedTimes(input, target, c.repeat)
			result = results[0]
			predict = predictFromResult(result)
			for _, v := range results {
				usedTime += int64(v.Time / time.Millisecond)
			}
		} else {
			for {
				result = m.Feed(input, target)
				predict = predictFromResult(result)
				usedTime += int64(result.Time / time.Millisecond)
				if uint8(predict) == data.Label {
					break
				}
			}
		}
		profile.Add(data.Label == uint8(predict))
		fmt.Printf("%5d: expect %d, predict %d, error %.4f, correct %.2f%%, recent %.2f%%, time %d ms\n",
			profile.Total, data.Label, predict, result.TotalError, profile.HitRate()*100, profile.RecentHitRate()*100, result.Time/time.Millisecond)
	}
	return c.saveModel(m)
}

func (c Context) Test() (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(c.checkData())

	m, err := c.loadModel()
	util.NoError(err)

	datas := mnist.Load(filepath.Join(c.dataPath, test_image), filepath.Join(c.dataPath, test_label), c.limit)

	profile := NewProfile(100)
	for {
		data, ok := <-datas
		if !ok {
			break
		}
		result := m.Test(mnistToData(data))
		predict := predictFromResult(result)
		profile.Add(data.Label == uint8(predict))
		fmt.Printf("%5d: expect %d, predict %d, error %.4f, correct %.2f%%, recent %.2f%%, time %d ms\n",
			profile.Total, data.Label, predict, result.TotalError, profile.HitRate()*100, profile.RecentHitRate()*100, result.Time/time.Millisecond)
		if c.savePath != "" && predict != int(data.Label) {
			util.NoError(data.SaveToFile(filepath.Join(c.savePath, fmt.Sprintf("%d-%d-%d.png", profile.Total, data.Label, predict))))
		}
	}
	return
}

func (c Context) checkData() error {
	if c.dataPath == "" {
		return errors.New("Data path not specified")
	}
	// TODO check MNIST files
	return nil
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
	} else {
		result.Init()
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
