package gortex

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

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
		t.Fatalf("h[0][0] must be 14 but %d.", h.Get(0, 0))
	}
	if h.Get(1, 0) != 32 {
		t.Fatalf("h[1][0] must be 32 but %d.", h.Get(1, 0))
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
		t.Fatalf("h[0][0] must be 15 but %d.", h.Get(0, 0))
	}
	if h.Get(1, 0) != 34 {
		t.Fatalf("h[1][0] must be 34 but %d.", h.Get(1, 0))
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
	s := NewSolver() // the Solver uses RMSProp

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
		s.Step(model, 1, 0.0, 0.0)
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
	target := 4

	// make optimizer
	s := NewSolver() // the Solver uses RMSProp

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
		s.Step(model, 0.1, 0.0, 0.0)
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

func TestOptimizationWithCrossentropySGD(t *testing.T) {
	// start from random
	rand.Seed(time.Now().UnixNano())
	// model W*x+b weights
	W := RandXavierMat(10, 4)   // weights Matrix
	b := RandXavierMat(10, 1)   // bias vector
	W1 := RandXavierMat(10, 10) // weights Matrix
	// random signal to map into target
	x := RandMat(4, 1) // input vector
	// target class for model - one out of 10
	target := 4

	// make optimizer
	s := NewSGDSolver() // the Solver uses SGD

	// update W and b, use learning rate of 0.01,
	// regularization strength of 0.0001 and clip gradient magnitudes at 5.0

	model := map[string]*Matrix{"W": W, "b": b, "W1": W1}
	// make 10 optimization steps
	for i := 0; i < 1000; i++ {
		G := Graph{NeedsBackprop: true}
		// make computation graph
		first := G.Tanh(G.Add(G.Mul(W, x), b))
		//first := G.Tanh(G.InstanceNormalization(G.Add(G.Mul(W, x), b)))
		output := G.Mul(W1, first)
		cost, probability := G.Crossentropy(output, target)
		t.Logf("step: %d crossentropy: %f perplexity: %f probability: %f\n", i, cost, math.Exp(float64(cost)), probability)
		if probability > 0.99 { // stop if we got desired result
			break
		}
		// compute gradients
		G.Backward()
		// update model weights
		s.Step(model, 0.03)
		// print error
	}
	G := Graph{}
	// make computation graph
	h := Softmax(G.Mul(W1, G.Tanh(G.Add(G.Mul(W, x), b))))
	t.Logf("vector of probabilities given signal x: %#v\n", h.W)
	if h.W[target] < 0.99 {
		t.Fatalf("model failed to optimize weights; prediction probability=%f must be very close to one", h.W[target])
	}

}

func TestDeltaRNN(t *testing.T) {
	// start from random
	rand.Seed(time.Now().UnixNano())
	trainFile := "ptb.train.txt"
	dic, e := DictionaryFromFile(trainFile, CharSplitter{})
	if e != nil {
		t.Fatal(e)
	}
	embedding_size := 128
	hidden_size := 128
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	s := NewSolver() // the Solver uses RMSPROP
	rnn := MakeRNN(embedding_size, hidden_size, dic.Len())
	//t.Logf("%s\n", rnn)
	LookupTable := RandXavierMat(embedding_size, dic.Len()) // Lookup Table matrix

	h0 := Mat(hidden_size, 1) // vector of zeros
	// define model parameters
	model := rnn.Model("RNN")
	model["LookupTable"] = LookupTable
	count := 0
	ma_ppl := NewMovingAverage(50)
	ma_nll := NewMovingAverage(50)
	ma_bpc := NewMovingAverage(50)
	//batch_size := 16

	SampleVisitor(trainFile, CharSplitter{}, dic, func(x []int) {
		if len(x) > 10 {
			// map term indexes in dictionary to embedding vectors
			G := Graph{NeedsBackprop: true}
			var yt *Matrix
			ht := h0
			var x_cost, x_probability float32
			//fmt.Printf("X:\n")
			// make forward through rnn through time
			for time, term_id := range x[:len(x)-1] {
				//fmt.Printf("%s", dic.TokenByID(term_id))
				xt := G.Lookup(LookupTable, term_id)

				ht, yt = rnn.Step(&G, xt, ht)
				// out task at each time step is to predict next symbol from rnn output
				cost, probability := G.Crossentropy(yt, x[time+1])
				x_cost += cost
				x_probability *= probability
				//t.Logf("step: %d crossentropy: %f perplexity: %f probability: %f\n", time, crossentropy, perplexity, probability)
			}
			//fmt.Printf("\n")
			x_cost /= float32(len(x) - 1)
			ma_bpc.Add(x_cost / math.Ln2)
			x_perplexity := float32(math.Exp(float64(x_cost)))
			ma_ppl.Add(x_perplexity)
			ma_nll.Add(x_cost)
			// compute gradients
			G.Backward()

			// update model weights
			//if count > 0 && count%batch_size == 0 {
			s.Step(model, 0.001, 0, 0)
			if count%10 == 0 {
				fmt.Printf("step: %d nll: %f perplexity: %f bpc: %f probability: %f\n", count, ma_nll.Avg(), ma_ppl.Avg(), ma_bpc.Avg(), x_probability)
			}

			count++
		}
		if count%100 == 0 { // print some model generated text
			fmt.Printf("MODEL GENERATED TEXT:\n")
			G := Graph{NeedsBackprop: false}
			ht := h0
			term_id := 10
			var logits *Matrix
			for i := 0; i < 100; i++ {
				xt := G.Lookup(LookupTable, term_id)
				ht, logits = rnn.Step(&G, xt, ht)
				term_id, _ = MaxIV(Softmax(logits))
				fmt.Printf("%s", dic.TokenByID(term_id))
				//t.Logf("step: %d crossentropy: %f perplexity: %f probability: %f\n", time, crossentropy, perplexity, probability)
			}
			fmt.Printf("\n")
		}
	})

}
