package assembler

import (
	"testing"
)

func l2Squared(x []float32) (sum float32) {
	for i := range x {
		sum += x[i] * x[i]
	}
	return
}

func TestL2Squared(t *testing.T) {
	vector2ScalarTest(L2Squared, l2Squared, t)
}

func BenchmarkL2Squared(b *testing.B) {
	vector2ScalarBench(l2Squared, b)
}

func BenchmarkOptimizedL2Squared(b *testing.B) {
	vector2ScalarBench(L2Squared, b)
}
