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
	if useSSE4 {
		funcVectorVector2scalarTest(SdotSSE4, sdot, t)
	}
}

func TestSdotAvx(t *testing.T) {
	if useAVX {
		funcVectorVector2scalarTest(SdotAvx, sdot, t)
	}
}

func BenchmarkSdot(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcVectorVector2scalarBench(sdot, b)
}

func BenchmarkSdotSSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useSSE4 {
		funcVectorVector2scalarBench(SdotSSE4, b)
	}
}

func BenchmarkSdotAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	if useAVX {
		funcVectorVector2scalarBench(SdotAvx, b)
	}
}
