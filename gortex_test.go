package gortex

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/vseledkin/gortex/assembler"
)

func TestDilatedConv(t *testing.T) {
	kernelSizes := []int{3,3}
	c := &DilatedTemporalConvolution{KernelSizes: kernelSizes, NumberOfKernels: []int{8,16}}
	x := make([]int, 11)

	for i := range x {
		x[i] = i
	}

	for l := range kernelSizes {
		for i := range x {
			field := c.ReceptiveField(i, l, x)
			t.Logf("layer: %d position: %d field: %+v", l, i, field)
		}
	}

}

func TestConv(t *testing.T) {
	k := 3
	s := 5
	l := 3
	x := make([]int, l)
	for i := range x {
		x[i] = i
	}

	o := (len(x)-k)/s + 1
	t.Logf("length: %d kernel: %d shift:%d outputs:%d", l, k, s, o)
	outputs := make([][]int, o)
	for i := range outputs {
		outputs[i] = x[i*s:s*i+k]
		t.Logf("%d %+v", i, outputs[i])
	}

	t.Logf("length=%d outputs=%d x = %+v", l, o, x)
	for i := range outputs {
		t.Logf("%d %+v", i, outputs[i])
	}
}

func TestMultinomial(t *testing.T) {
	n := 3
	p := RandMat(n, 1)
	p = Softmax(p)

	ps := p.ZerosAs()
	t.Logf("Probabilities %#v", p.W)
	for i := 0; i < 10000; i++ {
		sampled := Multinomial(p)
		ps.W[sampled]++
	}
	sum := assembler.Sum(ps.W)
	assembler.Sscale(1/sum, ps.W)
	t.Logf("Sampled Probabilities %#v", ps.W)
	// compare given and sampled probabilities
	for i := range p.W {
		diff := Abs(p.W[i] - ps.W[i])
		if diff > 0.01 {
			t.Fatalf("Given probability %f sampled %f diff = %f > 0.005", p.W[i], ps.W[i], diff)
		}
	}
}

func TestMatrixMul(t *testing.T) {
	W := MatFromSlice([][]float32{{1, 2, 3}, {4, 5, 6}}) // Matrix
	x := MatFromSlice([][]float32{{1}, {2}, {3}})        // input vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Mul(W, x)
	// the Graph structure keeps track of the connectivities between Mats
	t.Logf("%#v\n", h)

	if h.Rows != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.Rows)
	}
	if h.Columns != 1 {
		t.Fatalf("Must have 1 column but %d.", h.Columns)

	}
	if h.Get(0, 0) != 14 {
		t.Fatalf("h[0][0] must be 14 but %f.", h.Get(0, 0))
	}
	if h.Get(1, 0) != 32 {
		t.Fatalf("h[1][0] must be 32 but %f.", h.Get(1, 0))
	}
}

func TestMatrixMulAdd(t *testing.T) {
	W := MatFromSlice([][]float32{{1, 2, 3}, {4, 5, 6}}) // Matrix
	x := MatFromSlice([][]float32{{1}, {2}, {3}})        // input vector
	b := MatFromSlice([][]float32{{1}, {2}})             // bias vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Add(G.Mul(W, x), b)
	// the Graph structure keeps track of the connectivities between Mats
	t.Logf("%#v\n", h)

	if h.Rows != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.Rows)
	}
	if h.Columns != 1 {
		t.Fatalf("Must have 1 column but %d.", h.Columns)

	}
	if h.Get(0, 0) != 15 {
		t.Fatalf("h[0][0] must be 15 but %f.", h.Get(0, 0))
	}
	if h.Get(1, 0) != 34 {
		t.Fatalf("h[1][0] must be 34 but %f.", h.Get(1, 0))
	}

}

func TestOptimization(t *testing.T) {
	// start from random
	rand.Seed(time.Now().UnixNano())
	// model W*x+b weights
	W := RandXavierMat(10, 4) // weights Matrix
	b := RandXavierMat(10, 1) // bias vector
	// random signal to map into target
	x := RandMat(4, 1) // input vector
	// target for model
	target := MatFromSlice([][]float32{{1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}})

	// make optimizer
	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.01, Momentum: DefaultMomentum, Clip: 4})

	// update W and b, use learning rate of 0.02,
	// regularization strength of 0.0001 and clip gradient magnitudes at 5.0
	var mse float32
	model := map[string]*Matrix{"W": W, "b": b}
	// make 10 optimization steps
	for i := 0; i < 100; i++ {
		G := Graph{NeedsBackprop: true}
		// make computation graph
		mse = G.MSE(G.Add(G.Mul(W, x), b), target)
		// compute gradients
		G.Backward()
		// update model weights
		optimizer.Step(model)
		// print error
		t.Logf("step: %d err: %f\n", i, mse)
		if mse < 0.0001 {
			break
		}
	}
	G := Graph{}
	// make computation graph
	h := G.Add(G.Mul(W, x), b)
	t.Logf("vector mapped by learned function: %#v\n", h.W)
	if mse > 0.0001 {
		t.Fatalf("model failed to optimize weights; mse=%f but must be very close to zero", mse)
	}

}

func TestOptimizationWithCrossentropy1(t *testing.T) {
	// start from random
	rand.Seed(time.Now().UnixNano())
	// model W*x+b weights
	W := RandXavierMat(10, 4) // weights Matrix
	b := RandXavierMat(10, 1) // bias vector
	// random signal to map into target
	x := RandMat(4, 1) // input vector
	// target class for model - one out of 10
	target := uint(4)

	// make optimizer
	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.0003, Momentum: DefaultMomentum, Clip: 4})

	// update W and b, use learning rate of 0.01,
	// regularization strength of 0.0001 and clip gradient magnitudes at 5.0

	model := map[string]*Matrix{"W": W, "b": b}
	// make 10 optimization steps
	for i := 0; i < 1000; i++ {
		G := Graph{NeedsBackprop: true}
		// make computation graph
		cost, probability := G.Crossentropy(G.Add(G.Mul(W, x), b), target)
		// compute gradients
		G.Backward()
		// update model weights
		optimizer.Step(model)
		// print error
		t.Logf("step: %d crossentropy: %f perplexity: %f probability: %f\n", i, cost, math.Exp(float64(cost)), probability)
		if probability > 0.999 {
			break
		}
	}
	G := Graph{}
	// make computation graph
	h := Softmax(G.Add(G.Mul(W, x), b))
	t.Logf("vector of probabilities given signal x: %#v\n", h.W)
	if h.W[target] < 0.999 {
		t.Fatalf("model failed to optimize weights; prediction probability=%f must be very close to one", h.W[target])
	}

}

func TestGruRnnLanguageModel(t *testing.T) {
	// start from random
	rand.Seed(time.Now().UnixNano())
	trainFile := "input.txt"
	dic, e := DictionaryFromFile(trainFile, CharSplitter{})
	if e != nil {
		t.Fatal(e)
	}
	embedding_size := 128
	hidden_size := 128
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.001, Momentum: DefaultMomentum, Clip: 4})
	//rnn := MakeRNN(embedding_size, hidden_size, dic.Len())
	net := MakeGRU(embedding_size, hidden_size, dic.Len())
	//t.Logf("%s\n", rnn)
	LookupTable := RandMat(embedding_size, dic.Len()) // Lookup Table matrix

	h0 := Mat(hidden_size, 1) // vector of zeros
	// define model parameters
	model := net.GetParameters("RNN")
	model["LookupTable"] = LookupTable
	count := 0
	ma_ppl := NewMovingAverage(1000)
	ma_nll := NewMovingAverage(1000)
	ma_bpc := NewMovingAverage(1000)
	batch_size := 16

	anneal_rate := float32(0.9999)
	CharSampleVisitor(trainFile, 10, CharSplitter{}, dic, func(epoch int, x []uint) {
		// map term indexes in dictionary to embedding vectors
		var x_cost, x_probability float32
		//fmt.Printf("X:\n")

		// make forward through rnn through time
		G := &Graph{NeedsBackprop: true}
		ht := h0
		for i, term_id := range x[:len(x)-1] {
			var yt *Matrix
			ht, yt = net.Step(G, G.Lookup(LookupTable, int(term_id)), ht)

			// out task at each time step is to predict next symbol from rnn output
			cost, probability := G.Crossentropy(yt, x[i+1])

			x_cost += cost
			x_probability *= probability
		}
		G.Backward()

		x_cost /= float32(len(x) - 1)
		ma_bpc.Add(x_cost / math.Ln2)
		x_perplexity := float32(math.Exp(float64(x_cost)))
		ma_ppl.Add(x_perplexity)
		ma_nll.Add(x_cost)
		// compute gradients
		//for k, m := range model {
		//	fmt.Printf("%s %#v %f %f\n", k, m.DW[:2], m.Norm(), m.NormGradient())
		//}
		// update model weights
		count++
		if count > 0 && count%batch_size == 0 {
			//for k, m := range model {
			//	fmt.Printf("%s %#v %f %f\n", k, m.DW[:2], m.Norm(), m.NormGradient())
			//}
			//ScaleGradient(model, 1/float32(len(x)-1)/float32(batch_size))
			optimizer.Step(model)
			optimizer.LearningRate *= anneal_rate
		}

		if count%1000 == 0 { // print some model generated text
			fmt.Printf("step: %d nll: %f perplexity: %f bpc: %f lr: %f\n", count, ma_nll.Avg(), ma_ppl.Avg(), ma_bpc.Avg(), optimizer.LearningRate)
			fmt.Printf("MODEL GENERATED TEXT: ")
			G := Graph{NeedsBackprop: false}
			ht := RandMat(hidden_size, 1)
			term_id := uint(rand.Int31n(int32(dic.Len())))
			var logits *Matrix
			for i := 0; i < 100; i++ {
				xt := G.Lookup(LookupTable, int(term_id))
				ht, logits = net.Step(&G, xt, ht)
				//term_id, _ = MaxIV(Softmax(logits))
				term_id = Multinomial(Softmax(logits))
				token := dic.TokenByID(term_id)
				fmt.Printf("%s", token)
				if token == "." || token == "?" || token == "!" {
					break
				}
				//t.Logf("step: %d crossentropy: %f perplexity: %f probability: %f\n", time, crossentropy, perplexity, probability)
			}
			fmt.Printf("\n")
		}
	})
}

func TestMulticoreLSTMTraining(t *testing.T) {
	assembler.Init(true)
	// start from random
	rand.Seed(time.Now().UnixNano())
	trainFile := "input.txt"
	//trainFile := "64.unique.txt"
	modelName := "MultLSTM"
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

	embedding_size := 128
	hidden_size := 256
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.0003, Momentum: DefaultMomentum, Clip: 4})

	//rnn := MakeRNN(embedding_size, hidden_size, dic.Len())
	//rnn := MakeGRU(embedding_size, hidden_size, dic.Len())
	//net := MakeLSTM(embedding_size, hidden_size, dic.Len())
	net := MakeMultiplicativeLSTM(embedding_size, hidden_size, dic.Len())
	//net := MakeDeltaRNN(embedding_size, hidden_size, dic.Len())
	//net.ForgetGateTrick(2.0)
	//t.Logf("%s\n", rnn)

	h0 := Mat(hidden_size, 1) // vector of zeros
	// define model parameters
	var model map[string]*Matrix
	var LookupTable *Matrix
	if _, err := os.Stat(modelName); err == nil {
		model, e = LoadModel(modelName)
		if e != nil {
			t.Fatal(e)
		}
		e = net.SetParameters(modelName, model)
		if e != nil {
			t.Fatal(e)
		}
		LookupTable = model["LookupTable"]
	} else {
		model = net.GetParameters(modelName)
		LookupTable = RandMat(embedding_size, dic.Len()) // Lookup Table matrix
		model["LookupTable"] = LookupTable
	}

	count := 0
	ma_ppl := NewMovingAverage(1000)
	ma_nll := NewMovingAverage(1000)
	ma_bpc := NewMovingAverage(1000)
	ma_speed := NewMovingAverage(1000)
	ma_allspeed := NewMovingAverage(1000)
	ma_duration := NewMovingAverage(1000)
	batch_size := 32
	learning_rate := float32(0.001)
	anneal_rate := float32(0.9999)
	threads := batch_size
	type result struct {
		cost       float32
		perplexity float32
		bpc        float32
		time       time.Duration
		speed      float32
		len        int
	}
	license := make(chan interface{}, threads)
	results := make(chan *result, batch_size*threads)
	for i := 0; i < threads; i++ {
		license <- 1
	}
	go func() {
		var update_count, rnn_steps, samples_count int
		start := time.Now()
		for r := range results {
			update_count++
			samples_count++
			rnn_steps += r.len
			ma_bpc.Add(r.bpc)
			ma_ppl.Add(r.perplexity)
			ma_nll.Add(r.cost)
			ma_speed.Add(r.speed)
			ma_duration.Add(float32(r.time.Seconds()))
			if update_count == batch_size {
				for i := 0; i < threads; i++ { // freeze train
					<-license
				}
				ma_allspeed.Add(float32(rnn_steps) / float32(time.Now().Sub(start).Seconds()))
				update_count = 0
				ScaleGradient(model, 1/float32(rnn_steps))
				rnn_steps = 0
				optimizer.Step(model)
				fmt.Printf("step: %d nll: %f ppl: %f bpc: %f lr: %f speed:%f speed: %f time:%f s.\n", samples_count, ma_nll.Avg(), ma_ppl.Avg(), ma_bpc.Avg(), learning_rate, ma_speed.Avg(), ma_allspeed.Avg(), ma_duration.Avg())
				learning_rate = learning_rate * anneal_rate
				for i := 0; i < threads; i++ { // release train
					license <- 1
				}

				start = time.Now()

			}
			if samples_count%1000 == 0 {
				SaveModel(modelName, model)
			}
		}
	}()
	var w sync.WaitGroup
	w.Add(1)
	go func() {
		CharSampleVisitor(trainFile, 10, CharSplitter{}, dic, func(epoch int, x []uint) {

			<-license
			go func(x []uint) {
				start := time.Now()
				// map term indexes in dictionary to embedding vectors
				var x_cost, x_probability float32
				// make forward through rnn through time
				G := &Graph{NeedsBackprop: true}
				ht := h0
				ct := h0
				for i, term_id := range x[:len(x)-1] {
					var yt *Matrix
					ht, ct, yt = net.Step(G, G.Lookup(LookupTable, int(term_id)), ht, ct)
					//ht, yt = net.Step(G, G.Lookup(LookupTable, term_id), ht)
					// out task at each time step is to predict next symbol from rnn output
					cost, probability := G.Crossentropy(yt, x[i+1])
					x_cost += cost
					x_probability *= probability
				}
				G.Backward()

				x_cost /= float32(len(x) - 1)
				x_perplexity := float32(math.Exp(float64(x_cost)))
				duration := time.Now().Sub(start)
				results <- &result{
					cost:       x_cost,
					perplexity: x_perplexity,
					bpc:        x_cost / math.Ln2,
					time:       duration,
					speed:      float32(len(x)-1) / float32(duration.Seconds()),
					len:        len(x) - 1,
				}
				license <- 1
			}(x)

			count++
			if count%100 == 0 { // print some model generated text
				fmt.Printf("MODEL GENERATED TEXT: ")
				G := Graph{NeedsBackprop: false}
				ht := RandMat(hidden_size, 1)
				ct := RandMat(hidden_size, 1)
				term_id := uint(rand.Int31n(int32(dic.Len())))
				var logits *Matrix
				for i := 0; i < 100; i++ {
					xt := G.Lookup(LookupTable, int(term_id))
					//ct.W[88] = 1
					ht, ct, logits = net.Step(&G, xt, ht, ct)
					//ht, logits = net.Step(&G, xt, ht)
					//if term_id == dic.IDByToken(" ") {
					term_id = Multinomial(Softmax(logits))
					//} else {
					//	term_id, _ = MaxIV(Softmax(logits))
					//}
					fmt.Printf("%s", dic.TokenByID(term_id))
				}
				fmt.Printf("\n")

				if count%100 == 0 { // print some model generated text
					f, e := os.Create("dynamics.html")
					if e != nil {
						t.Fatal()
					}
					f.WriteString(
						`<!DOCTYPE html>
								<html lang="en">
									<head>
    									<meta charset="UTF-8">
    								</head>
    								<body>
    								<table cellspacing="0"><tr>
`)
					G := Graph{NeedsBackprop: false}
					ht := h0
					ct := h0
					rows := make([]string, hidden_size)
					var logits *Matrix
					for i, term_id := range x[:len(x)-1] {
						ht, ct, logits = net.Step(&G, G.Lookup(LookupTable, int(term_id)), ht, ct)
						probs := Softmax(logits)
						term := dic.TokenByID(term_id)
						if term == " " {
							term = "_"
						}
						max_term_id, _ := MaxIV(probs)
						for ai, _ := range probs.W {
							color := ""
							if max_term_id == x[i+1] {
								color = fmt.Sprintf("rgb(0,%d,0)", int(256*probs.W[max_term_id]))
							} else {
								color = fmt.Sprintf("rgb(%d,0,0)", int(-256*probs.W[max_term_id]))
							}
							if len(rows[ai]) == 0 {
								rows[ai] += fmt.Sprintf(`<td>%d</td>`, ai)
							}
							rows[ai] += fmt.Sprintf(`<td style="font-weight:bold;background-color:%s">%s</td>`, color, term)
						}
					}
					// write terms
					for _, row := range rows {
						f.WriteString("<tr>")
						f.WriteString(row)
						f.WriteString("</tr>")
					}
					f.WriteString(
						`</tr></table>
						</body>
						</html>
`)
					f.Close()
				}
			}
		})
	}()
	w.Wait()
	for i := 0; i < threads; i++ {
		<-license
	}
}

func BenchmarkSoftmax(b *testing.B) {
	b.StopTimer()
	assembler.Init(false)
	x := RandMat(1000, 1)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Softmax(x)
	}

}

func BenchmarkOptimizedSoftmax(b *testing.B) {
	b.StopTimer()
	assembler.Init(true)
	x := RandMat(1000, 1)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Softmax(x)
	}
}

func TestAutoencoder(t *testing.T) {
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "input.txt"
	dic, e := DictionaryFromFile(trainFile, CharSplitter{})
	if e != nil {
		t.Fatal(e)
	}
	hidden_size := 128
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.0003, Momentum: DefaultMomentum, Clip: 4})

	encoder := MakeGRU(dic.Len(), hidden_size, 1)
	decoder := MakeGRU(hidden_size, hidden_size, dic.Len())

	// define model parameters
	encoderModel := encoder.GetParameters("Encoder")
	decoderModel := decoder.GetParameters("Decoder")

	count := 0
	ma_d := NewMovingAverage(50)
	//batch_size := 8

	max_len := 32

	var e_steps, d_steps float32
	learning_rate := float32(0.001)
	anneal_rate := float32(0.999)

	CharSampleVisitor(trainFile, 10, CharSplitter{}, dic, func(epoch int, x []uint) {
		if len(x) > max_len {
			return
		}
		// read sample
		sample := ""
		for i := range x {
			sample += dic.TokenByID(x[i])
		}
		G := &Graph{NeedsBackprop: true}
		ht := Mat(hidden_size, 1) // vector of zeros
		var z *Matrix
		// encode sequence into z
		for i := range x {
			e_steps++
			oneHot := Mat(dic.Len(), 1)
			oneHot.W[x[i]] = 1.0
			ht, _ = encoder.Step(G, oneHot, ht)
		}
		z = ht // this is the last state of encoder
		// decode sequence from z
		ht = Mat(hidden_size, 1) // vector of zeros
		var logit *Matrix
		cost := float32(0)

		decoded := ""
		for i := range x {
			d_steps++
			ht, logit = decoder.Step(G, z, ht)
			c, _ := G.Crossentropy(logit, x[i])
			cid, _ := MaxIV(Softmax(logit))
			decoded += dic.TokenByID(cid)
			cost += c
		}
		cost /= float32(len(x))
		G.Backward()
		ScaleGradient(encoderModel, 1/e_steps)
		ScaleGradient(decoderModel, 1/d_steps)
		optimizer.Step(encoderModel)
		optimizer.Step(decoderModel)
		d_steps = 0
		e_steps = 0

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
			fmt.Printf("z: %#v\n", z.W[:10])
			learning_rate = learning_rate * anneal_rate
			//if avg_cost < 0.0001 {
			//	max_len++
			//	ma_d.Add(1)
			//	fmt.Printf("max len ++: %d\n", max_len)
			//}
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
