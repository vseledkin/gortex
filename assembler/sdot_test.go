package assembler

import (
	"testing"
)



func sdot(X, Y []float32) (dot float32) {
	for i := range X {
		dot += X[i] * Y[i]
	}
	return
}

func TestSdotSSE4(t *testing.T) {
	funcVectorVector2scalarTest(SdotSSE4, sdot, t)
}

func TestSdotAvx(t *testing.T) {
	funcVectorVector2scalarTest(SdotAvx, sdot, t)
}

func BenchmarkSdot(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVector2scalarBench(sdot, b)
}

func BenchmarkSdotSSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVector2scalarBench(SdotSSE4, b)
}

func BenchmarkSdotAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVector2scalarBench(SdotAvx, b)
}
