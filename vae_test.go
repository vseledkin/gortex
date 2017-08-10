package gortex

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/vseledkin/gortex/assembler"
	"github.com/gizak/termui"
)

func TestVae(t *testing.T) {

		// maintain random seed
		rand.Seed(time.Now().UnixNano())
		trainFile := "input.txt"
		dic, e := CharDictionaryFromFile(trainFile, CharSplitter{})
		if e != nil {
			t.Fatal(e)
		}
		hidden_size := 128
		embedding_size := 128
		z_size := 128
		fmt.Printf("Dictionary has %d tokens\n", dic.Len())
		fmt.Printf("%s\n", dic)

		s := NewSolver()                                  // the Solver uses RMSPROP
		LookupTable := RandMat(embedding_size, dic.Len()) // Lookup Table matrix
		encoder := MakeOutputlessLSTM(embedding_size, hidden_size)
		encoder.ForgetGateTrick(2.0)
		vae := MakeVae(hidden_size, z_size)
		decoder := MakeLSTM(z_size, hidden_size, dic.Len())
		decoder.ForgetGateTrick(2.0)

		// define model parameters
		encoderModel := encoder.GetParameters("Encoder")
		encoderModel["LookupTable"] = LookupTable
		vaeModel := vae.GetParameters("VAE")
		decoderModel := decoder.GetParameters("Decoder")

		count := 0
		ma_d := NewMovingAverage(50)
		//batch_size := 8

		max_len := 32

		var e_steps, d_steps float32
		learning_rate := float32(0.001)
		anneal_rate := float32(0.999)
		batch_size := 16
		CharSampleVisitor(trainFile, 10, CharSplitter{}, dic, func(x []uint) {
			if len(x) > max_len {
				return
			}
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
			distribution, mean, dev := vae.Step(G, ct)
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
			if count % batch_size == 0 && count > 0 {
				//ScaleGradient(encoderModel, 1/e_steps)
				//ScaleGradient(decoderModel, 1/d_steps)
				s.Step(encoderModel, learning_rate, 0.0, 5.0)
				s.Step(vaeModel, learning_rate, 0.0, 0.0)
				s.Step(decoderModel, learning_rate, 0.0, 5.0)
				d_steps = 0
				e_steps = 0
			}
			count++
			//if count > 0 && count%batch_size == 0 {
			//d_cost /= d_steps
			//g_cost /= g_steps
			ma_d.Add(cost)
			//if sample != decoded {
			//}
			avg_cost := ma_d.Avg()
			if count%150 == 0 {
				fmt.Printf("\ndecoded: [%s]\n", decoded)
				fmt.Printf("encoded: [%s]\n", sample)
				fmt.Printf("step: %d loss: %f lr: %f\n", count, avg_cost, learning_rate)
				fmt.Printf("dev: %f mean: %f\n", assembler.L1(dev.W), assembler.L1(mean.W))
				fmt.Printf("dev: %#v\n", dev.W[:10])
				mid, miv := MaxIV(dev)
				fmt.Printf("dev: %d %f\n", mid, miv)
				learning_rate = learning_rate * anneal_rate
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

	termui.Loop()
}
