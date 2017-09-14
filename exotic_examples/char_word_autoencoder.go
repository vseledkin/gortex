package main

import (
	"time"
	"fmt"
	"os"
	"math/rand"
	g "github.com/vseledkin/gortex"
	"log"
)

func main(){
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "input.txt"
	modelName := "WordCharAE"
	dic, e := g.LoadDictionary(modelName + ".dic")
	if e != nil {
		dic, e = g.DictionaryFromFile(trainFile, g.CharSplitter{})
		if e != nil {
			log.Fatal(e)
		}
		e = g.SaveDictionary(modelName+".dic", dic)
		if e != nil {
			log.Fatal(e)
		}
	}

	hidden_size := 128
	embedding_size := 128
	z_size := 128
	fmt.Printf("%s\n", dic)
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())

	optimizer := g.NewOptimizer(g.OpOp{Method: g.WINDOWGRAD, LearningRate: 0.0003, Momentum: g.DefaultMomentum, Clip: 4})
	LookupTable := g.RandMat(embedding_size, dic.Len()) // Lookup Table matrix
	word_encoder := g.MakeOutputlessGRU(embedding_size, hidden_size)
	word_encoder.ForgetGateTrick(2.0)

	word_decoder := g.MakeGRU(z_size, hidden_size, dic.Len())
	word_decoder.ForgetGateTrick(2.0)

	model := make(map[string]*g.Matrix)
	// define model parameters
	for k, v := range word_encoder.GetParameters("Encoder") {
		model[k] = v
	}
	model["LookupTable"] = LookupTable

	for k, v := range word_decoder.GetParameters("Decoder") {
		model[k] = v
	}

	if _, err := os.Stat(modelName); err == nil {
		loadedModel, e := g.LoadModel(modelName)
		if e != nil {
			log.Fatal(e)
		}
		e = g.SetParameters(model, loadedModel)
		if e != nil {
			log.Fatal(e)
		}
	}

	count := 0
	ma_cost := g.NewMovingAverage(512)

	var e_steps, d_steps float32

	batch_size := 32

	threads := 4
	license := make(chan struct{}, threads)
	for i := 0; i < threads; i++ {
		license <- struct{}{}
	}

	g.WordCharSampleVisitor(trainFile, g.CharSplitter{}, dic, func(epoch int, x [][]uint) {
		<-license
		count++
		go func(count int) {
			// read sample
			sample := ""
			for i := range x {
				sample += dic.TokenByID(x[i])
			}
			G := &g.Graph{NeedsBackprop: true}
			ht :=g.Mat(hidden_size, 1).OnesAs() // vector of zeros

			// encode sequence into z
			for i := range x {
				e_steps++
				embedding := G.Lookup(LookupTable, int(x[i]))
				ht = word_encoder.Step(G, embedding, ht)
			}
			z := ht
			// decode sequence from z
			var logit *g.Matrix
			cost := float32(0)

			decoded := ""
			ht = g.Mat(hidden_size, 1).OnesAs() // vector of zeros
			for i := range x {
				d_steps++
				ht, logit = word_decoder.Step(G, z, ht)
				c, _ := G.Crossentropy(logit, x[i])
				cid, _ := g.MaxIV(g.Softmax(logit))
				decoded += dic.TokenByID(cid)
				cost += c
			}
			cost /= float32(len(x))
			G.Backward()

			if count%batch_size == 0 && count > 0 {
				//ScaleGradient(encoderModel, 1/e_steps)
				//ScaleGradient(decoderModel, 1/d_steps)
				optimizer.Step(model)
				d_steps = 0
				e_steps = 0
			}
			count++

			ma_cost.Add(cost)
			avg_cost := ma_cost.Avg()

			if count%10000 == 0 {
				g.SaveModel(modelName, model)
			}

			if count%500 == 0 {

				fmt.Printf("\ndecoded: [%s]\n", decoded)
				fmt.Printf("encoded: [%s]\n", sample)
				fmt.Printf("epoch: %d step: %d loss: %f lr: %f\n", epoch, count, avg_cost, optimizer.LearningRate)

			}
			license <- struct{}{}
		}(count)
	})
	for i := 0; i < threads; i++ {
		<-license
	}
}