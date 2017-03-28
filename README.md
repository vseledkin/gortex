# gortex

Pure go neural network library

## Example code
    
    Features: 
        autodifferentiation
        rnn models support
        language model sample (in test)
## Example code
```go
    // see gortex_test.go
    // learn function to map from any to particular vector

    // start from random
	rand.Seed(time.Now().UnixNano())
	// model W*x+b weights
	W := RandMat(10, 4) // weights Matrix
	b := RandMat(10, 1) // bias vector
	// random signal to map into target
	x := RandMat(4, 1) // input vector
	// target for model
	target := MatFromSlice([][]float64{{1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}});

	// make optimizer
	s := NewSolver() // the Solver uses RMSProp

	// update W and b, use learning rate of 0.02,
	// regularization strength of 0.0001 and clip gradient magnitudes at 5.0
	var mse float64

	// model weights to optimize
	model := map[string]*Matrix{"W":W, "b":b}
	// make 10 optimization steps
	for i := 0; i < 10; i++ {
		G := Graph{NeedsBackprop:true}
		// make computation graph
		mse = G.MSE(G.Add(G.Mul(W, x), b), target)
		// compute gradients
		G.Backward()
		// update model weights
		s.Step(model, 0.02, 0.0001, 5.0)
		// print error
		t.Logf("step: %d err: %f\n", i, mse)
	}
	G := Graph{}
	// test resulting function
	h := G.Add(G.Mul(W, x), b)
	t.Logf("Vector mapped by learned function: %#v\n", h.W)
	if mse > 0.001 {
		t.Fatalf("model failed to optimize weights of the model mse=%f but must be very close to zero", mse)
	}
```
## Warning: Beta

Very beta




