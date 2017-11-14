package assembler

import (
	"testing"
)

func sum(x []float32) float32 {
	var s float32
	for i := range x {
		s += x[i]
	}
	return s
}

func TestSum(t *testing.T) {
	vector2ScalarTest(Sum, sum, t)
}

func BenchmarkSum(b *testing.B) {
	vector2ScalarBench(sum, b)
}

func BenchmarkOptimizedSum(b *testing.B) {
	vector2ScalarBench(Sum, b)
}
