package assembler

import (
	"testing"
)

func sxpy(X, Y []float32) {
	for i := range X {
		Y[i] += X[i]
	}
}

func TestSxpySSE4(t *testing.T) {
	if useSSE4 {
		funcVectorVectorTest(SxpySSE4, sxpy, t)
	}
}

func TestSxpyAvx(t *testing.T) {
	if useAVX {
		funcVectorVectorTest(SxpyAvx, sxpy, t)
	}
}

func BenchmarkSxpy(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVectorBench(sxpy, b)
}

func BenchmarkSxpySSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useSSE4 {
		funcVectorVectorBench(SxpySSE4, b)
	}
}

func BenchmarkSxpyAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useAVX {
		funcVectorVectorBench(SxpyAvx, b)
	}
}
