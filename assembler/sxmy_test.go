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
	funcVectorVectorTest(SxmySSE4, sxmy, t)
}

func TestSxmyAvx(t *testing.T) {
	funcVectorVectorTest(SxmyAvx, sxmy, t)
}

func BenchmarkSxmy(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVectorBench(sxmy, b)
}

func BenchmarkSxmySSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVectorBench(SxmySSE4, b)
}

func BenchmarkSxmyAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVectorBench(SxmyAvx, b)
}
