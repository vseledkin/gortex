package assembler

import (
	"testing"
	"log"
)

func init() {
	var a float32 = 2
	x := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	y := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	//log.Printf("avx %f", SaxpyAvx(a, x, y))
	SaxpyAvx(a, x, y)
	log.Printf("x %v", x)

	//log.Printf("sse4 %f", SaxpySSE4(a, x, y))

	//log.Printf("generic %F", saxpy(a, x, y))
}

func saxpy(a float32, X []float32, Y []float32) {
	for i := range X {
		Y[i] += a * X[i]
	}
}

func TestSaxpySSE4(t *testing.T) {
	funcScalarVectorVectorTest(SaxpySSE4, saxpy, t)
}

func TestSaxpyAvx(t *testing.T) {
	funcScalarVectorVectorTest(SaxpyAvx, saxpy, t)
}

func BenchmarkSaxpy(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcScalarVectorVectorBench(saxpy, b)
}

func BenchmarkSaxpySSE4Optimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcScalarVectorVectorBench(SaxpySSE4, b)
}

func BenchmarkSaxpyAvxOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	funcScalarVectorVectorBench(SaxpyAvx, b)
}
