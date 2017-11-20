package assembler

import "testing"

func sxmuley(X, Y []float32) {
	for i := range X {
		Y[i] *= X[i]
	}
}

func TestSxmuley(t *testing.T) {
	vector2VectorTest(Sxmuley, sxmuley, t)
}

func BenchmarkSxmuley(b *testing.B) {
	vector2VectorBench(sxmuley, b)
}

func BenchmarkOptimizedSxmuley(b *testing.B) {
	vector2VectorBench(Sxmuley, b)
}
