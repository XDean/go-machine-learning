package main

import (
	"errors"
	"fmt"
	core2 "github.com/XDean/go-machine-learning/ann/core"
	"github.com/XDean/go-machine-learning/ann/util"
	"github.com/XDean/go-machine-learning/sample/mnist"
	"path/filepath"
	"strings"
	"time"
)

type Context struct {
	fullDesc bool
	loadPath string
	savePath string
	dataPath string
	limit    int
	modelN   int
	batch    int
}

func (c Context) Show() error {
	for i, v := range models {
		s := ""
		if c.fullDesc {
			s = v.Full()
		} else {
			s = v.Brief()
		}
		sb := &strings.Builder{}
		index := fmt.Sprintf("%d. ", i+1)
		sb.WriteString(index)
		util.WriteWithPrefix(sb, s, strings.Repeat(" ", len(index)))
		fmt.Println(sb.String())
	}
	return nil
}

func (c Context) Train() (err error) {
	defer util.RecoverNoError(&err)
	util.NoError(c.checkData())

	m, err := c.loadModel()
	util.NoError(err)

	resultStream := m.FeedBatch(
		adapt(mnist.Load(filepath.Join(c.dataPath, train_image), filepath.Join(c.dataPath, train_label), c.limit)),
		c.batch)
	profile := NewProfile(100)
	startTime := time.Now()
	for {
		result, ok := <-resultStream
		if !ok {
			break
		}
		used := time.Since(startTime)
		expect := expectFromResult(result)
		predict := predictFromResult(result)
		profile.Add(expect == predict)
		fmt.Printf("%5d: expect %d, predict %d, error %.4f, correct %.2f%%, recent %.2f%%, total time %d ms\n",
			profile.Total, expect, predict, result.TotalError, profile.HitRate()*100, profile.RecentHitRate()*100, used/time.Millisecond)
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
		startTime := time.Now()
		result := m.Test(mnistToData(data))
		used := time.Since(startTime)
		predict := predictFromResult(result)
		profile.Add(data.Label == uint8(predict))
		fmt.Printf("%5d: expect %d, predict %d, error %.4f, correct %.2f%%, recent %.2f%%, time %d ms\n",
			profile.Total, data.Label, predict, result.TotalError, profile.HitRate()*100, profile.RecentHitRate()*100, used/time.Millisecond)
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

func (c Context) loadModel() (result *core2.Model, err error) {
	defer func() {
		if err == nil {
			fmt.Println(result.Full())
		}
	}()
	if c.modelN > 0 && c.modelN <= len(models) {
		result = models[c.modelN-1]
		fmt.Printf("Use built-in model: %s\n", result.Name)
	}
	if c.loadPath != "" {
		fmt.Printf("Load model from %s\n", c.loadPath)
		result = new(core2.Model)
		err = result.LoadFromFile(c.loadPath)
	}
	if result == nil {
		err = errors.New("Model is not specified.")
	}
	return
}

func (c Context) saveModel(m *core2.Model) (err error) {
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
