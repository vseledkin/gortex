package gortex

import (
	"math/rand"
	"testing"
	"time"
)

func TestMatrixMul(t *testing.T) {
	W := MatFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}}) // Matrix
	x := MatFromSlice([][]float64{{1}, {2}, {3}});       // input vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Mul(W, x)
	// the Graph structure keeps track of the connectivities between Mats
	t.Logf("%#v\n", h)

	if h.n != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.n)
	}
	if h.d != 1 {
		t.Fatalf("Must have 1 column but %d.", h.d)

	}
	if h.Get(0, 0) != 14 {
		t.Fatalf("h[0][0] must be 14 but %d.", h.Get(0, 0))
	}
	if h.Get(1, 0) != 32 {
		t.Fatalf("h[1][0] must be 32 but %d.", h.Get(1, 0))
	}
}

func TestMatrixMulAdd(t *testing.T) {
	W := MatFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}}) // Matrix
	x := MatFromSlice([][]float64{{1}, {2}, {3}});       // input vector
	b := MatFromSlice([][]float64{{1}, {2}});            // bias vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Add(G.Mul(W, x), b)
	// the Graph structure keeps track of the connectivities between Mats
	t.Logf("%#v\n", h)

	if h.n != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.n)
	}
	if h.d != 1 {
		t.Fatalf("Must have 1 column but %d.", h.d)

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
	// make computation graph
	h := G.Add(G.Mul(W, x), b)
	t.Logf("vector mapped by learned function: %#v\n", h.W)
	if mse > 0.001 {
		t.Fatalf("model failed to optimize weights of the model mse=%f but must be very close to zero", mse)
	}

}
