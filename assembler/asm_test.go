package assembler

import (
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
