package gortex

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/vseledkin/gortex/assembler"
)

func reverse(ss []uint) {
	last := len(ss) - 1
	for i := 0; i < len(ss)/2; i++ {
		ss[i], ss[last-i] = ss[last-i], ss[i]
	}
}

func TestCharRnnVae(t *testing.T) {
	words := false
	delim := ""

	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "top.txt"
	modelName := "CharRnnVAE"
	if words {
		modelName = "WordRnnVAE"
		delim = " "
	}
	dic, e := LoadDictionary(modelName + ".dic")
	if e != nil {
		if words {
			dic, e = DictionaryFromFile(trainFile, WordSplitter{})
		} else {
			dic, e = DictionaryFromFile(trainFile, CharSplitter{})
		}
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

	//kld_scale := float32(0.0)
	optimizer := NewOptimizer(OpOp{Method: ADAM, LearningRate: 0.0007, Momentum: DefaultMomentum, Clip: 7})
	optimizer.Iteration = 1000
	LookupTable := RandMatMD(embedding_size, dic.Len(), 0, 0.1) // Lookup Table matrix
	LookupTableCounts := make(map[int]int)
	encoder := MakeOutputlessGRU(embedding_size, hidden_size)

	vae := MakeVae(hidden_size, z_size)
	decoder := MakeGRU(hidden_size+dic.Len(), hidden_size, dic.Len())

	model := make(map[string]*Matrix)
	// define model parameters
	encoderModel := encoder.GetParameters("Encoder")
	for k, v := range encoderModel {
		model[k] = v
	}
	model["LookupTable"] = LookupTable
	vaeModel := vae.GetParameters("VAE")
	for k, v := range vaeModel {
		model[k] = v
	}
	decoderModel := decoder.GetParameters("Decoder")
	for k, v := range decoderModel {
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

	var e_steps, d_steps float32

	batch_size := 16
	threads := 1
	license := make(chan struct{}, threads)
	for i := 0; i < threads; i++ {
		license <- struct{}{}
	}

	kld_scale := float32(0.00001)
	epsilon := 1.0
	xxx := float32(0)
	xxx_step := float32(0.05)
	missCount := float32(0)
	gotCount := float32(0)
	start_annealing := false
	CharSampleVisitor(trainFile, 1, CharSplitter{}, dic, func(epoch int, x []uint) {
		//WordSampleVisitor(trainFile, WordSplitter{}, dic, func(epoch int, x []uint) {
		if len(x) == 0 {
			return
		}
		<-license
		count++
		go func(count int) {
			// read sample
			sample := ""
			for i := range x {
				sample += dic.TokenByID(x[i]) + delim
			}
			G := &Graph{NeedsBackprop: true, NeedsParallel: false}
			ht := Mat(hidden_size, 1).OnesAs() // vector of zeros

			// encode sequence into z
			reverse(x)
			for i := range x {
				e_steps++
				embedding := G.Lookup2(LookupTable, int(x[i]), LookupTableCounts)
				ht = encoder.Step(G, embedding, ht)
			}
			distribution, mean, logvar := vae.StepAmplitude(G, ht, epsilon)
			// estimate KLD
			kld := vae.KLD(G, kld_scale, mean, logvar)
			// decode sequence from z
			cost := float32(0)

			decoded := ""
			distribution = vae.Step1(G, distribution)
			ht = Mat(hidden_size, 1).OnesAs()
			logit := Mat(dic.Len(), 1)
			logit.W[dic.IDByToken(" ")] = 0.89
			ht = Mat(hidden_size, 1).OnesAs()
			reverse(x)
			for i := range x {
				d_steps++

				ht, logit = decoder.Step(G, G.Concat(distribution, G.Softmax(logit)), ht)

				c, _ := G.Crossentropy(logit, x[i])
				cid, _ := MaxIV(Softmax(logit))
				decoded += dic.TokenByID(cid) + delim
				cost += c
				if cid == x[i] {
					gotCount++
				} else {
					missCount++
				}
			}
			cost = cost / float32(len(x))

			if math.IsNaN(float64(cost)) {
				panic("NAN cost detected!!")
			}
			//println(cost)
			G.Backward()

			if count%batch_size == 0 && count > 0 {
				mutex.Lock()

				for i, count := range LookupTableCounts {
					if count > 1 {
						assembler.Sscale(1/float32(count), LookupTable.DW[i*LookupTable.Rows:i*LookupTable.Rows+LookupTable.Rows])
					}
					if count > 0 {
						LookupTableCounts[i] = 0
					}
				}

				ScaleGradient(encoderModel, 1/e_steps)
				ScaleGradient(decoderModel, 1/d_steps)
				ScaleGradient(vaeModel, 1/d_steps)
				optimizer.Step(model)
				mutex.Unlock()
				d_steps = 0
				e_steps = 0
			}
			count++

			ma_kld_cost.Add(kld)
			ma_cost.Add(cost)

			if count%100000 == 0 {
				mutex.Lock()
				SaveModel(modelName, model)
				mutex.Unlock()
			}

			if count%1000 == 0 {

				avg_cost := ma_cost.Avg()
				dev := Exp(logvar)

				fmt.Printf("\nopit: %f\n", optimizer.Iteration)
				fmt.Printf("decoded: [%s]\n", decoded)
				fmt.Printf("encoded: [%s]\n", sample)
				mg := missCount / gotCount
				fmt.Printf("epoch: %d step: %d loss: %f lr: %f kld_scale: %.10f epsilon: %.10f m/g=%f\n", epoch, count, avg_cost, optimizer.LearningRate, kld_scale, epsilon, mg)
				fmt.Printf("kld: %f\n", ma_kld_cost.Avg())
				fmt.Printf("dev : %#v\n", dev.W[:10])
				fmt.Printf("mean: %#v\n", mean.W[:10])
				gotCount = 0
				missCount = 0
				if !start_annealing {
					if avg_cost < 0.5 {
						start_annealing = true
					}
				}
				if start_annealing {
					kld_scale = Sigmoid(10, xxx)
					xxx += xxx_step
				}
				optimizer.LearningRate *= 0.9995

				// interpolate between two pints
				z1 := RandMatMD(vae.z_size, 1, 0, epsilon)
				z2 := RandMat(vae.z_size, 1)
				gg := &Graph{NeedsBackprop: false}

				fmt.Printf("Interpolation\n")
				for a := float32(0.0); a <= 1; a += 0.1 {
					z := gg.Add(gg.MulConstant(1.0-a, z1), gg.MulConstant(a, z2))
					decoded := ""
					z = vae.Step1(gg, z)
					ht = Mat(hidden_size, 1).OnesAs()
					logit := Mat(dic.Len(), 1)
					logit.W[dic.IDByToken(" ")] = 0.89
				loop:
					for range make([]struct{}, 32) {
						ht, logit = decoder.Step(gg, gg.Concat(z, gg.Softmax(logit)), ht)
						//cid, _ := MaxIV(Softmax(logit))
						cid := Multinomial(Softmax(logit))
						token := dic.TokenByID(cid)
						decoded += token + delim
						switch token {
						case ".", "!", "?", "…":
							break loop
						}
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
