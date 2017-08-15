package gortex

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/vseledkin/gortex/assembler"
)

func TestBabiChallemge(t *testing.T) {
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "100000_32.txt"
	modelName := "CharVAE"
	dic, e := LoadDictionary(modelName + ".dic")
	if e != nil {
		dic, e = DictionaryFromFile(trainFile, CharSplitter{})
		if e != nil {
			t.Fatal(e)
		}
		e = SaveDictionary(modelName+".dic", dic)
		if e != nil {
			t.Fatal(e)
		}
	}

	hidden_size := 128
	embedding_size := 128
	z_size := 128
	fmt.Printf("%s\n", dic)
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())

	s := NewSolver()                                  // the Solver uses RMSPROP
	LookupTable := RandMat(embedding_size, dic.Len()) // Lookup Table matrix
	encoder := MakeOutputlessLSTM(embedding_size, hidden_size)
	encoder.ForgetGateTrick(2.0)
	vae := MakeVae(hidden_size, z_size)
	decoder := MakeLSTM(z_size, hidden_size, dic.Len())
	decoder.ForgetGateTrick(2.0)

	model := make(map[string]*Matrix)
	// define model parameters
	for k, v := range encoder.GetParameters("Encoder") {
		model[k] = v
	}
	model["LookupTable"] = LookupTable
	for k, v := range vae.GetParameters("VAE") {
		model[k] = v
	}

	for k, v := range decoder.GetParameters("Decoder") {
		model[k] = v
	}

	if _, err := os.Stat(modelName); err == nil {
		loadedModel, e := LoadModel(modelName)
		if e != nil {
			t.Fatal(e)
		}
		e = SetParameters(model, loadedModel)
		if e != nil {
			t.Fatal(e)
		}
	}

	count := 0
	ma_cost := NewMovingAverage(512)
	ma_kld_cost := NewMovingAverage(512)
	ma_mean := NewMovingAverage(512)
	ma_dev := NewMovingAverage(512)
	//batch_size := 8

	var e_steps, d_steps float32
	learning_rate := float32(0.001)

	batch_size := 16
	kld_scale := float32(0.000001)
	CharSampleVisitor(trainFile, 1, CharSplitter{}, dic, func(epoch int, x []uint) {
		// read sample
		sample := ""
		for i := range x {
			sample += dic.TokenByID(x[i])
		}
		G := &Graph{NeedsBackprop: true}
		ht := Mat(hidden_size, 1).OnesAs() // vector of zeros
		ct := Mat(hidden_size, 1).OnesAs() // vector of zeros

		// encode sequence into z
		for i := range x {
			e_steps++
			embedding := G.Lookup(LookupTable, int(x[i]))
			ht, ct = encoder.Step(G, embedding, ht, ct)
		}
		distribution, mean, logvar := vae.Step(G, ct)
		// estimate KLD
		kld := vae.KLD(G, kld_scale, mean, logvar)
		// decode sequence from z
		var logit *Matrix
		cost := float32(0)

		decoded := ""
		ht = Mat(hidden_size, 1).OnesAs() // vector of zeros
		ct = Mat(hidden_size, 1).OnesAs() // vector of zeros
		for i := range x {
			d_steps++
			ht, ct, logit = decoder.Step(G, distribution, ht, ct)
			c, _ := G.Crossentropy(logit, x[i])
			cid, _ := MaxIV(Softmax(logit))
			decoded += dic.TokenByID(cid)
			cost += c
		}
		cost /= float32(len(x))
		G.Backward()
		if count%batch_size == 0 && count > 0 {
			//ScaleGradient(encoderModel, 1/e_steps)
			//ScaleGradient(decoderModel, 1/d_steps)
			s.Step(model, learning_rate, 0.00001, 5.0)
			d_steps = 0
			e_steps = 0
		}
		count++
		//if count > 0 && count%batch_size == 0 {
		//d_cost /= d_steps
		//g_cost /= g_steps
		ma_kld_cost.Add(kld)
		ma_cost.Add(cost)
		m := assembler.Sum(mean.W) / float32(len(mean.W))
		dev := Exp(logvar)
		v := assembler.Sum(dev.W) / float32(len(dev.W))
		ma_mean.Add(m)
		ma_dev.Add(float32(math.Sqrt(float64(v))))
		//if sample != decoded {
		//}
		avg_cost := ma_cost.Avg()
		avg_mean := ma_mean.Avg()
		avg_dev := ma_dev.Avg()

		if count%10000 == 0 {
			SaveModel(modelName, model)
		}

		if count%500 == 0 {

			fmt.Printf("\ndecoded: [%s]\n", decoded)
			fmt.Printf("encoded: [%s]\n", sample)
			fmt.Printf("epoch: %d step: %d loss: %f lr: %f kld_scale: %f\n", epoch, count, avg_cost, learning_rate, kld_scale)
			fmt.Printf("mean: %f dev: %f kld: %f\n", avg_mean, avg_dev, ma_kld_cost.Avg())
			fmt.Printf("dev: %#v\n", dev.W[:10])

			if avg_cost < 0.2 && kld_scale < 1.0 {
				f, e := os.Open("kld_ch.json")
				if e != nil {
					t.Error(e)
				}
				cob, e := ioutil.ReadAll(f)
				if e != nil {
					t.Error(e)
				}
				var co struct{ GateInc, Lr float32 }
				json.Unmarshal(cob, &co)
				f.Close()
				kld_scale += co.GateInc
				learning_rate = co.Lr
			}
			// interpolate between two pints
			z1 := RandMat(vae.z_size, 1)
			z2 := RandMat(vae.z_size, 1)
			gg := &Graph{NeedsBackprop: false}

			fmt.Printf("Interpolation\n")
			for a := float32(0.0); a <= 1; a += 0.1 {
				z := gg.Add(gg.MulConstant(1.0-a, z1), gg.MulConstant(a, z2))
				decoded := ""
				ht = Mat(hidden_size, 1).OnesAs() // vector of zeros
				ct = Mat(hidden_size, 1).OnesAs() // vector of zeros
				for range make([]struct{}, 32) {
					ht, ct, logit = decoder.Step(gg, z, ht, ct)
					cid, _ := MaxIV(Softmax(logit))
					decoded += dic.TokenByID(cid)
				}
				fmt.Printf("%0.2f sentence: %s\n", a, decoded)
			}
		}
		/*
			if count%100 == 0 { // print some model generated text
				learning_rate = learning_rate * anneal_rate
				// sample noise
				//z := RandMat(z_size, 1)
				fmt.Printf("MODEL GENERATED TEXT: ")
				G := Graph{NeedsBackprop: false}
				ht := Mat(g_hidden_size, 1) // vector of zeros
				var logit *Matrix
				for i := 0; i < max_len; i++ {
					ht, logit = generator.Step(&G, z, ht)
					term_id, _ := MaxIV(Softmax(logit))
					fmt.Printf("%s", dic.TokenByID(term_id))
				}
				fmt.Printf("\n")
			}*/
	})
}
