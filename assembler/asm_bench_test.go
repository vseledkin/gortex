package assembler

import (
	"math/rand"
	"testing"
)

func BenchmarkSscale(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sscale(float32(rand.NormFloat64()), x)
	}
}

func BenchmarkOptimizedSscale(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sscale(float32(rand.NormFloat64()), x)
	}
}
