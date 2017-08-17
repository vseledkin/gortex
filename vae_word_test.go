package gortex

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"math"

	"encoding/json"
	"github.com/vseledkin/gortex/assembler"
	"io/ioutil"
	"os"
)

func TestWordVae(t *testing.T) {
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "100000_32.txt"
	modelName := "WordVAE"
	dic, e := LoadDictionary(modelName + ".dic")
	if e != nil {
		dic, e = DictionaryFromFile(trainFile, WordSplitter{})
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

	//optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.001, Momentum: DefaultMomentum, Clip: 5})
	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.001, Momentum: DefaultMomentum, Clip: 4})
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

	batch_size := 32
	kld_scale := float32(0.0001)
	threads := 4
	license := make(chan struct{}, threads)
	for i := 0; i < threads; i++ {
		license <- struct{}{}
	}

	WordSampleVisitor(trainFile, WordSplitter{}, dic, func(epoch int, x []uint) {
		<-license
		count++
		go func(count int) {
			// read sample
			sample := ""
			for i := range x {
				sample += dic.TokenByID(x[i]) + " "
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
			distribution, mean, logvar := vae.Step(G, ht)
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
				decoded += dic.TokenByID(cid) + " "
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
				fmt.Printf("epoch: %d step: %d loss: %f lr: %f kld_scale: %f\n", epoch, count, avg_cost, optimizer.LearningRate, kld_scale)
				fmt.Printf("mean: %f dev: %f kld: %f\n", avg_mean, avg_dev, ma_kld_cost.Avg())
				fmt.Printf("dev: %#v\n", dev.W[:10])

				if avg_cost < 1 && kld_scale < 1.0 {
					f, e := os.Open("kld.json")
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
					optimizer.LearningRate = co.Lr
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
						decoded += dic.TokenByID(cid) + " "
					}
					fmt.Printf("%0.2f sentence: %s\n", a, decoded)
				}
			}
			license <- struct{}{}
		}(count)
	})
	for i := 0; i < threads; i++ {
		<-license
	}
}
