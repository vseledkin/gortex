package assembler

import (
	"testing"
)

func sscale(a float32, X []float32) {
	for i := range X {
		X[i] *= a
	}
}

func TestSscale(t *testing.T) {
	scalarVector2VectorTest(Sscale, sscale, t)
}

func BenchmarkSscale(b *testing.B) {
	scalarVector2VectorBench(sscale, b)
}

func BenchmarkOptimizedSscale(b *testing.B) {
	scalarVector2VectorBench(Sscale, b)
}
