package gortex

import (
	"fmt"
	"testing"
)

func TestMatrixMul(t *testing.T) {
	W := MatFromSlice([][]float64{{1,2,3},{4,5,6}})    // Matrix
	x := MatFromSlice([][]float64{{1},{2},{3}});  // input vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Mul(W, x)
	// the Graph structure keeps track of the connectivities between Mats
	fmt.Printf("%#v\n",h)

	if h.n != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.n)
	}
	if h.d != 1 {
		t.Fatalf("Must have 1 column but %d.", h.d)

	}
	if h.Get(0,0) != 14 {
		t.Fatalf("h[0][0] must be 14 but %d.", h.Get(0,0))
	}
	if h.Get(1,0) != 32 {
		t.Fatalf("h[1][0] must be 32 but %d.", h.Get(1,0))
	}
}

func TestMatrixMulAdd(t *testing.T) {
	W := MatFromSlice([][]float64{{1,2,3},{4,5,6}})    // Matrix
	x := MatFromSlice([][]float64{{1},{2},{3}});  // input vector
	b := MatFromSlice([][]float64{{1},{2}});  // bias vector

	// matrix multiply followed by bias offset. h is a Mat
	G := new(Graph)
	h := G.Add(G.Mul(W, x),b)
	// the Graph structure keeps track of the connectivities between Mats
	fmt.Printf("%#v\n",h)

	if h.n != 2 {
		t.Fatalf("Must have 2 rows but %d.", h.n)
	}
	if h.d != 1 {
		t.Fatalf("Must have 1 column but %d.", h.d)

	}
	if h.Get(0,0) != 15 {
		t.Fatalf("h[0][0] must be 15 but %d.", h.Get(0,0))
	}
	if h.Get(1,0) != 34 {
		t.Fatalf("h[1][0] must be 34 but %d.", h.Get(1,0))
	}

}


func TestGraph(t *testing.T) {
	W := RandMat(10, 4)    // weights Matrix
	x := RandMat(4, 1)  // input vector
	b := RandMat(10, 1) // bias vector

	// matrix multiply followed by bias offset. h is a Mat
	G := Graph{NeedsBackprop:true}

	h := G.Add(G.Mul(W, x), b)
	// the Graph structure keeps track of the connectivities between Mats
	fmt.Printf("%#v\n",h)
	// we can now set the loss on h
	h.dw[0] = 1.0 // say we want the first value to be lower

	// propagate all gradients backwards through the graph
	// starting with h, all the way down to W,x,b
	// i.e. this sets .dw field for W,x,b with the gradients
	G.Backward()
	fmt.Printf("h: %#v\n",h.dw)
	fmt.Printf("b: %#v\n",b.dw)
	fmt.Printf("x: %#v\n",x.dw)
	fmt.Printf("W: %#v\n",W.w)

	// do a parameter update on W,b:
	s := NewSolver() // the Solver uses RMSProp
	fmt.Printf("solver: %#v\n",s)
	// update W and b, use learning rate of 0.01,
	// regularization strength of 0.0001 and clip gradient magnitudes at 5.0
	model := map[string]*Matrix{"W":W, "b":b}
	s.Step(model, 0.01, 0.0001, 5.0)
	fmt.Printf("W: %#v\n",W.w)
}
