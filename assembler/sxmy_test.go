package assembler

import (
	"testing"
)

func sxmy(X, Y []float32) {
	for i := range X {
		Y[i] *= X[i]
	}
}

func TestSxmySSE4(t *testing.T) {
	if useSSE4 {
		funcVectorVectorTest(SxmySSE4, sxmy, t)
	}
}

func TestSxmyAvx(t *testing.T) {
	if useAVX {
		funcVectorVectorTest(SxmyAvx, sxmy, t)
	}
}

func BenchmarkSxmy(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVectorBench(sxmy, b)
}

func BenchmarkSxmySSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useSSE4 {
		funcVectorVectorBench(SxmySSE4, b)
	}
}

func BenchmarkSxmyAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useAVX {
		funcVectorVectorBench(SxmyAvx, b)
	}
}
