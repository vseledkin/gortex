package assembler

import (
	"math"
	"testing"
)

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func TestSqrtSquared(t *testing.T) {
	scalarTest(Sqrt, sqrt, t)
}

func BenchmarkSqrtSquared(b *testing.B) {
	scalarBench(sqrt, b)
}

func BenchmarkOptimizedSqrtSquared(b *testing.B) {
	scalarBench(Sqrt, b)
}
