package assembler

import (
	"math"
	"math/rand"
	"testing"
)

func TestSscale(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		X1 := make([]float32, j)
		Y := make([]float32, j)
		for i := range X {
			X[i] = float32(rand.NormFloat64())
			X1[i] = X[i]
		}
		factor := float32(rand.NormFloat64())
		sscale(factor, X)
		Sscale(factor, X1)
		for i := range Y {
			if X[i] != X1[i] {
				t.Fatalf("sums do not match want %f got %f in vector of length %d\n", X[i], X1[i], len(X))
			}
		}
	}
}

func TestSsum(t *testing.T) {
	for j := 0; j < 100; j++ {
		vector := make([]float32, j)
		for i := range vector {
			vector[i] = float32(rand.NormFloat64())
		}
		sumfast := Ssum(vector)
		sum := ssum(vector)
		// numeric issues
		if float32(math.Abs(float64(sumfast-sum))) > 1e-5 {
			t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", sum, sumfast, len(vector))
		}
	}
}
