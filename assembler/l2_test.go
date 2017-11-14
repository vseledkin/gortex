package assembler

import (
	"testing"
)

func l2(x []float32) (sum float32) {
	for i := range x {
		sum += x[i] * x[i]
	}
	sum = Sqrt(sum)
	return
}

func TestL2(t *testing.T) {
	vector2ScalarTest(L2, l2, t)
}

func BenchmarkL2(b *testing.B) {
	vector2ScalarBench(l2, b)
}

func BenchmarkOptimizedL2(b *testing.B) {
	vector2ScalarBench(L2, b)
}
